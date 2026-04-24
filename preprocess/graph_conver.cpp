#include <atomic>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <memory>
#include <array>
#include <omp.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#include <intrin.h>
#include <xmmintrin.h>
#else
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif

#ifdef _MSC_VER
#define PFETCH(p) _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T0)
#elif defined(__GNUC__) || defined(__clang__)
#define PFETCH(p) __builtin_prefetch((p), 0, 3)
#else
#define PFETCH(p) ((void)0)
#endif

#if defined(__GNUC__) && !defined(__clang__) && defined(_OPENMP) && !defined(_WIN32)
#include <parallel/algorithm>
#define PSORT_GNU 1
#endif
#if !defined(PSORT_GNU) && defined(__has_include)
#if __has_include(<execution>)
#include <execution>
#define PSORT_STL 1
#endif
#endif

namespace fs = std::filesystem;
using uint = unsigned int;
using u64  = unsigned long long;

struct Edge { uint u, v; };

class MMapFile {
public:
    explicit MMapFile(const std::string& path) {
#ifdef _WIN32
        hFile_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                             nullptr, OPEN_EXISTING,
                             FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
                             nullptr);
        if (hFile_ == INVALID_HANDLE_VALUE)
            throw std::runtime_error("Cannot open input: " + path);
        LARGE_INTEGER sz;
        if (!GetFileSizeEx(hFile_, &sz)) {
            CloseHandle(hFile_);
            throw std::runtime_error("GetFileSizeEx failed: " + path);
        }
        fsize_ = (size_t)sz.QuadPart;
        if (fsize_ == 0) { data_ = nullptr; return; }
        hMap_ = CreateFileMappingA(hFile_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hMap_) { CloseHandle(hFile_); throw std::runtime_error("CreateFileMapping failed"); }
        data_ = (const char*)MapViewOfFile(hMap_, FILE_MAP_READ, 0, 0, 0);
        if (!data_) { CloseHandle(hMap_); CloseHandle(hFile_); throw std::runtime_error("MapViewOfFile failed"); }
#else
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("Cannot open input: " + path);
        struct stat sb{};
        if (::fstat(fd_, &sb) != 0) { ::close(fd_); throw std::runtime_error("fstat failed"); }
        fsize_ = (size_t)sb.st_size;
        if (fsize_ == 0) { data_ = nullptr; return; }
        void* p = ::mmap(nullptr, fsize_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (p == MAP_FAILED) { ::close(fd_); throw std::runtime_error("mmap failed"); }
        data_ = (const char*)p;
        ::madvise(p, fsize_, MADV_SEQUENTIAL);
        ::madvise(p, fsize_, MADV_WILLNEED);
#endif
    }

    ~MMapFile() {
#ifdef _WIN32
        if (data_) UnmapViewOfFile(data_);
        if (hMap_) CloseHandle(hMap_);
        if (hFile_ != INVALID_HANDLE_VALUE) CloseHandle(hFile_);
#else
        if (data_) ::munmap((void*)data_, fsize_);
        if (fd_ >= 0) ::close(fd_);
#endif
    }

    MMapFile(const MMapFile&) = delete;
    MMapFile& operator=(const MMapFile&) = delete;

    const char* data() const noexcept { return data_; }
    size_t size() const noexcept { return fsize_; }

private:
    const char* data_ = nullptr;
    size_t fsize_ = 0;

#ifdef _WIN32
    HANDLE hFile_ = INVALID_HANDLE_VALUE;
    HANDLE hMap_ = nullptr;
#else
    int fd_ = -1;
#endif
};

class MMapWriter {
public:
    MMapWriter(const std::string& path, size_t size) : size_(size) {
        if (size == 0) {
            FILE* fp = std::fopen(path.c_str(), "wb");
            if (!fp) throw std::runtime_error("Cannot create output: " + path);
            std::fclose(fp);
            return;
        }
#ifdef _WIN32
        hFile_ = CreateFileA(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                             CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hFile_ == INVALID_HANDLE_VALUE)
            throw std::runtime_error("Cannot create output: " + path);
        LARGE_INTEGER li; li.QuadPart = (LONGLONG)size;
        if (!SetFilePointerEx(hFile_, li, nullptr, FILE_BEGIN) || !SetEndOfFile(hFile_)) {
            CloseHandle(hFile_);
            throw std::runtime_error("SetEndOfFile failed");
        }
        hMap_ = CreateFileMappingA(hFile_, nullptr, PAGE_READWRITE, 0, 0, nullptr);
        if (!hMap_) { CloseHandle(hFile_); throw std::runtime_error("CreateFileMapping(write) failed"); }
        data_ = (char*)MapViewOfFile(hMap_, FILE_MAP_WRITE, 0, 0, 0);
        if (!data_) { CloseHandle(hMap_); CloseHandle(hFile_); throw std::runtime_error("MapViewOfFile(write) failed"); }
#else
        fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (fd_ < 0) throw std::runtime_error("Cannot create output: " + path);
        if (::ftruncate(fd_, (off_t)size) != 0) { ::close(fd_); throw std::runtime_error("ftruncate failed"); }
        void* p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (p == MAP_FAILED) { ::close(fd_); throw std::runtime_error("mmap(write) failed"); }
        data_ = (char*)p;
#endif
    }

    ~MMapWriter() {
#ifdef _WIN32
        if (data_) { FlushViewOfFile(data_, 0); UnmapViewOfFile(data_); }
        if (hMap_) CloseHandle(hMap_);
        if (hFile_ != INVALID_HANDLE_VALUE) CloseHandle(hFile_);
#else
        if (data_) { ::msync(data_, size_, MS_SYNC); ::munmap(data_, size_); }
        if (fd_ >= 0) ::close(fd_);
#endif
    }

    MMapWriter(const MMapWriter&) = delete;
    MMapWriter& operator=(const MMapWriter&) = delete;

    char* data() noexcept { return data_; }
    size_t size() const noexcept { return size_; }

private:
    char* data_ = nullptr;
    size_t size_ = 0;

#ifdef _WIN32
    HANDLE hFile_ = INVALID_HANDLE_VALUE;
    HANDLE hMap_ = nullptr;
#else
    int fd_ = -1;
#endif
};

template <typename T>
class ParallelArray {
public:
    ParallelArray() = default;

    explicit ParallelArray(size_t n) : size_(n) {
        if (n) {
            data_ = (T*)std::malloc(n * sizeof(T));
            if (!data_) throw std::bad_alloc();
        }
    }

    ~ParallelArray() { reset(); }

    ParallelArray(const ParallelArray&) = delete;
    ParallelArray& operator=(const ParallelArray&) = delete;

    ParallelArray(ParallelArray&& o) noexcept : data_(o.data_), size_(o.size_) {
        o.data_ = nullptr;
        o.size_ = 0;
    }

    ParallelArray& operator=(ParallelArray&& o) noexcept {
        if (this != &o) {
            reset();
            data_ = o.data_;
            size_ = o.size_;
            o.data_ = nullptr;
            o.size_ = 0;
        }
        return *this;
    }

    void resize(size_t n) {
        reset();
        size_ = n;
        if (n) {
            data_ = (T*)std::malloc(n * sizeof(T));
            if (!data_) throw std::bad_alloc();
        }
    }

    void parallel_fill(const T& v) {
#pragma omp parallel for schedule(static)
        for (long long i = 0; i < (long long)size_; ++i) data_[i] = v;
    }

    void first_touch() {
#pragma omp parallel for schedule(static)
        for (long long i = 0; i < (long long)size_; ++i) data_[i] = T{};
    }

    void reset() {
        if (data_) {
            std::free(data_);
            data_ = nullptr;
            size_ = 0;
        }
    }

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }

    T& operator[](size_t i) noexcept { return data_[i]; }
    const T& operator[](size_t i) const noexcept { return data_[i]; }

    size_t size() const noexcept { return size_; }

    T* begin() noexcept { return data_; }
    T* end() noexcept { return data_ + size_; }
    const T* begin() const noexcept { return data_; }
    const T* end() const noexcept { return data_ + size_; }

private:
    T* data_ = nullptr;
    size_t size_ = 0;
};

template <typename In, typename Out>
static void parallel_exclusive_scan(const In* in, Out* out, size_t N) {
    int nth = omp_get_max_threads();
    if (nth <= 1 || N < 2048) {
        out[0] = Out(0);
        Out r = 0;
        for (size_t i = 0; i < N; ++i) {
            r += Out(in[i]);
            out[i + 1] = r;
        }
        return;
    }

    std::vector<Out> partial((size_t)nth + 1, Out(0));
    out[0] = Out(0);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int anth = omp_get_num_threads();
        size_t chunk = (N + (size_t)anth - 1) / (size_t)anth;
        size_t beg = std::min((size_t)tid * chunk, N);
        size_t end = std::min(beg + chunk, N);

        Out s = 0;
        for (size_t i = beg; i < end; ++i) s += Out(in[i]);
        partial[tid + 1] = s;

#pragma omp barrier
#pragma omp single
        {
            for (int i = 1; i <= anth; ++i) partial[i] += partial[i - 1];
        }

        Out r = partial[tid];
        for (size_t i = beg; i < end; ++i) {
            r += Out(in[i]);
            out[i + 1] = r;
        }
    }
}

static void parallel_radix_sort_u32(uint32_t* data, size_t n) {
    if (n < 2) return;

    int nth = omp_get_max_threads();
    if (nth <= 1 || n < 65536) {
        std::sort(data, data + n);
        return;
    }

    ParallelArray<uint32_t> tmp(n);
    tmp.first_touch();

    constexpr int RADIX = 256;
    std::vector<std::array<size_t, RADIX>> hist(nth);
    std::vector<std::array<size_t, RADIX>> offs(nth);

    uint32_t* src = data;
    uint32_t* dst = tmp.data();

    for (int pass = 0; pass < 4; ++pass) {
        int shift = pass * 8;
        for (auto& h : hist) h.fill(0);

#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int anth = omp_get_num_threads();
            size_t chunk = (n + (size_t)anth - 1) / (size_t)anth;
            size_t beg = std::min((size_t)tid * chunk, n);
            size_t end = std::min(beg + chunk, n);
            auto& h = hist[tid];
            for (size_t i = beg; i < end; ++i) ++h[(src[i] >> shift) & 0xFFu];
        }

        size_t total = 0;
        for (int b = 0; b < RADIX; ++b) {
            for (int t = 0; t < nth; ++t) {
                size_t c = hist[t][b];
                offs[t][b] = total;
                total += c;
            }
        }

#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int anth = omp_get_num_threads();
            size_t chunk = (n + (size_t)anth - 1) / (size_t)anth;
            size_t beg = std::min((size_t)tid * chunk, n);
            size_t end = std::min(beg + chunk, n);
            auto o = offs[tid];
            for (size_t i = beg; i < end; ++i) {
                uint32_t x = src[i];
                dst[o[(x >> shift) & 0xFFu]++] = x;
            }
        }

        std::swap(src, dst);
    }
}

static size_t parallel_unique_sorted_u32(uint32_t* data, size_t n) {
    if (n < 2) return n;

    int nth = omp_get_max_threads();
    if (nth <= 1 || n < 65536)
        return (size_t)(std::unique(data, data + n) - data);

    std::vector<uint8_t> keep(n);
    keep[0] = 1;

#pragma omp parallel for schedule(static, 4096)
    for (long long i = 1; i < (long long)n; ++i)
        keep[i] = (data[i] != data[i - 1]) ? 1 : 0;

    std::vector<size_t> pos(n + 1);
    parallel_exclusive_scan(keep.data(), pos.data(), n);
    size_t uc = pos[n];

    ParallelArray<uint32_t> tmp(uc);
    tmp.first_touch();

#pragma omp parallel for schedule(static, 4096)
    for (long long i = 0; i < (long long)n; ++i)
        if (keep[i]) tmp[pos[i]] = data[i];

#pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)uc; ++i) data[i] = tmp[i];

    return uc;
}

template <typename It, typename Cmp>
static inline void psort(It first, It last, Cmp cmp) {
#if defined(PSORT_GNU)
    __gnu_parallel::sort(first, last, cmp);
#elif defined(PSORT_STL)
    std::sort(std::execution::par_unseq, first, last, cmp);
#else
    std::sort(first, last, cmp);
#endif
}

static inline void insertion_sort_u32(uint* b, uint* e) {
    for (uint* i = b + 1; i < e; ++i) {
        uint v = *i;
        uint* j = i;
        while (j > b && j[-1] > v) {
            j[0] = j[-1];
            --j;
        }
        *j = v;
    }
}

static inline uint* sort_unique_u32(uint* b, uint* e) {
    ptrdiff_t n = e - b;
    if (n <= 1) return e;
    if (n <= 32) insertion_sort_u32(b, e);
    else std::sort(b, e);
    uint* wr = b + 1;
    for (uint* rd = b + 1; rd < e; ++rd)
        if (*rd != wr[-1]) *wr++ = *rd;
    return wr;
}

static inline uint fast_atoi(const char*& p, const char* pend) {
    while (p < pend && (*p < '0' || *p > '9')) ++p;
    uint val = 0;
    while (p < pend && *p >= '0' && *p <= '9') {
        val = val * 10u + uint(*p - '0');
        ++p;
    }
    return val;
}

static inline uint degree_bucket(uint deg) {
    uint b = 0;
    u64 limit = 32ull;
    while ((u64)deg > limit && b < 8) {
        ++b;
        limit *= 32ull;
    }
    return b;
}

struct CliArgs {
    std::string infile, outfile;
    bool help = false;
};

static bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " -i <input> -o <output_prefix|output.graph.bin>" << std::endl;
}

static bool parse_args(int argc, char* argv[], CliArgs& out) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need_val = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << std::endl;
                return nullptr;
            }
            return argv[++i];
        };

        if (a == "-i") {
            const char* v = need_val("-i");
            if (!v) return false;
            out.infile = v;
        }
        else if (a == "-o") {
            const char* v = need_val("-o");
            if (!v) return false;
            out.outfile = v;
        }
        else if (a == "-h" || a == "--help") {
            out.help = true;
        }
        else {
            std::cerr << "Unknown option: " << a << std::endl;
            return false;
        }
    }
    return true;
}

void Convert(const std::string& infile, std::string outfile) {
    if (!ends_with(outfile, ".graph.bin")) outfile += ".graph.bin";

    fs::path out_path(outfile);
    if (out_path.has_parent_path()) fs::create_directories(out_path.parent_path());

    const int nthreads = omp_get_max_threads();
    omp_set_num_threads(nthreads);

    MMapFile mmf(infile);
    const char* data = mmf.data();
    size_t fsize = mmf.size();
    if (fsize == 0 || data == nullptr) {
        std::cerr << "Input is empty: " << infile << std::endl;
        std::exit(1);
    }

    std::vector<std::vector<Edge>> tl_edges(nthreads);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        size_t chunk = fsize / nthreads;
        size_t beg = (size_t)tid * chunk;
        size_t end = (tid == nthreads - 1) ? fsize : beg + chunk;

        if (tid != 0 && beg > 0)
            while (beg < fsize && data[beg - 1] != '\n') ++beg;
        if (tid != nthreads - 1 && end > 0)
            while (end < fsize && data[end - 1] != '\n') ++end;

        auto& buf = tl_edges[tid];
        buf.reserve((end - beg) / 12 + 16);

        const char* p = data + beg;
        const char* pend = data + end;
        while (p < pend) {
            if (*p == '\n' || *p == '\r' || *p == ' ' || *p == '\t') {
                ++p;
                continue;
            }
            if (*p == '%' || *p == '#') {
                while (p < pend && *p != '\n') ++p;
                continue;
            }
            uint u = fast_atoi(p, pend);
            uint v = fast_atoi(p, pend);
            buf.push_back({u, v});
            while (p < pend && *p != '\n') ++p;
        }
    }

    std::vector<uint> eoff(nthreads + 1, 0);
    for (int i = 0; i < nthreads; ++i)
        eoff[i + 1] = eoff[i] + (uint)tl_edges[i].size();
    uint total_edges = eoff[nthreads];

    ParallelArray<Edge> raw_edges(total_edges);
#pragma omp parallel for
    for (int i = 0; i < nthreads; ++i) {
        std::memcpy(raw_edges.data() + eoff[i],
                    tl_edges[i].data(),
                    tl_edges[i].size() * sizeof(Edge));
    }
    std::vector<std::vector<Edge>>().swap(tl_edges);

    ParallelArray<uint> id_buf((size_t)total_edges * 2);
#pragma omp parallel for schedule(static, 4096)
    for (long long i = 0; i < (long long)total_edges; ++i) {
        id_buf[2 * i]     = raw_edges[i].u;
        id_buf[2 * i + 1] = raw_edges[i].v;
    }

    parallel_radix_sort_u32(id_buf.data(), id_buf.size());
    uint total_nodes = (uint)parallel_unique_sorted_u32(id_buf.data(), id_buf.size());

    const uint* sorted_ids = id_buf.data();
#pragma omp parallel for schedule(static, 4096)
    for (long long i = 0; i < (long long)total_edges; ++i) {
        raw_edges[i].u = (uint)(std::lower_bound(sorted_ids, sorted_ids + total_nodes,
                                                 raw_edges[i].u) - sorted_ids);
        raw_edges[i].v = (uint)(std::lower_bound(sorted_ids, sorted_ids + total_nodes,
                                                 raw_edges[i].v) - sorted_ids);
    }
    id_buf.reset();

    std::vector<uint8_t> seen_src(total_nodes, 0);
    std::vector<uint8_t> seen_dst(total_nodes, 0);
#pragma omp parallel for schedule(static, 4096)
    for (long long i = 0; i < (long long)total_edges; ++i) {
        seen_src[raw_edges[i].u] = 1;
        seen_dst[raw_edges[i].v] = 1;
    }

    std::vector<uint> src_prefix(total_nodes + 1);
    std::vector<uint> dst_prefix(total_nodes + 1);
    parallel_exclusive_scan(seen_src.data(), src_prefix.data(), (size_t)total_nodes);
    parallel_exclusive_scan(seen_dst.data(), dst_prefix.data(), (size_t)total_nodes);

    uint nLeft  = src_prefix[total_nodes];
    uint nRight = dst_prefix[total_nodes];
    uint N = nLeft + nRight;

    std::vector<uint> src_map(total_nodes);
    std::vector<uint> dst_map(total_nodes);
#pragma omp parallel for schedule(static, 4096)
    for (long long i = 0; i < (long long)total_nodes; ++i) {
        src_map[i] = seen_src[i] ? src_prefix[i] : (uint)-1;
        dst_map[i] = seen_dst[i] ? dst_prefix[i] : (uint)-1;
    }
    std::vector<uint8_t>().swap(seen_src);
    std::vector<uint8_t>().swap(seen_dst);
    std::vector<uint>().swap(src_prefix);
    std::vector<uint>().swap(dst_prefix);

    std::vector<uint> degrees(N, 0);
    bool tls_ok = ((u64)nthreads * N * sizeof(uint)) < (u64(4) << 30);
    std::vector<std::vector<uint>> tl_deg;

    if (tls_ok) {
        tl_deg.resize(nthreads);
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            uint chunk = total_edges / (uint)nthreads;
            uint beg = (uint)tid * chunk;
            uint end = (tid == nthreads - 1) ? total_edges : beg + chunk;
            tl_deg[tid].assign(N, 0);
            uint* ld = tl_deg[tid].data();
            for (uint i = beg; i < end; ++i) {
                ++ld[src_map[raw_edges[i].u]];
                ++ld[dst_map[raw_edges[i].v] + nLeft];
            }
        }
#pragma omp parallel for schedule(static, 1024)
        for (long long i = 0; i < (long long)N; ++i) {
            uint acc = 0;
            for (int t = 0; t < nthreads; ++t) {
                uint d = tl_deg[t][i];
                tl_deg[t][i] = acc;
                acc += d;
            }
            degrees[i] = acc;
        }
    } else {
        std::vector<std::atomic<uint>> adeg(N);
#pragma omp parallel for
        for (long long i = 0; i < (long long)N; ++i)
            adeg[i].store(0, std::memory_order_relaxed);

#pragma omp parallel for schedule(static, 4096)
        for (long long i = 0; i < (long long)total_edges; ++i) {
            adeg[src_map[raw_edges[i].u]].fetch_add(1, std::memory_order_relaxed);
            adeg[dst_map[raw_edges[i].v] + nLeft].fetch_add(1, std::memory_order_relaxed);
        }

#pragma omp parallel for
        for (long long i = 0; i < (long long)N; ++i)
            degrees[i] = adeg[i].load(std::memory_order_relaxed);
    }

    std::vector<uint> offset(N + 1);
    parallel_exclusive_scan(degrees.data(), offset.data(), (size_t)N);
    uint total_entries = offset[N];

    ParallelArray<uint> neighbors(total_entries);
    neighbors.first_touch();

    if (tls_ok) {
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            uint chunk = total_edges / (uint)nthreads;
            uint beg = (uint)tid * chunk;
            uint end = (tid == nthreads - 1) ? total_edges : beg + chunk;
            uint* my_pos = tl_deg[tid].data();

            for (uint i = beg; i < end; ++i) {
                uint u = src_map[raw_edges[i].u];
                uint v = dst_map[raw_edges[i].v] + nLeft;
                neighbors[offset[u] + my_pos[u]++] = v;
                neighbors[offset[v] + my_pos[v]++] = u;
            }
        }
        std::vector<std::vector<uint>>().swap(tl_deg);
    } else {
        std::vector<std::atomic<uint>> pos(N);
#pragma omp parallel for
        for (long long i = 0; i < (long long)N; ++i)
            pos[i].store(0, std::memory_order_relaxed);

#pragma omp parallel for schedule(static, 4096)
        for (long long i = 0; i < (long long)total_edges; ++i) {
            uint u = src_map[raw_edges[i].u];
            uint v = dst_map[raw_edges[i].v] + nLeft;
            uint pu = pos[u].fetch_add(1, std::memory_order_relaxed);
            uint pv = pos[v].fetch_add(1, std::memory_order_relaxed);
            neighbors[offset[u] + pu] = v;
            neighbors[offset[v] + pv] = u;
        }
    }

    raw_edges.reset();
    std::vector<uint>().swap(src_map);
    std::vector<uint>().swap(dst_map);

    std::vector<uint> new_degrees(N, 0);
#pragma omp parallel for schedule(dynamic, 256)
    for (long long i = 0; i < (long long)N; ++i) {
        uint* b = neighbors.data() + offset[i];
        uint* e = neighbors.data() + offset[i + 1];
        uint* ne = sort_unique_u32(b, e);
        new_degrees[i] = (uint)(ne - b);
    }

    auto sort_bucketed = [&](uint start, uint count, bool is_left) -> std::vector<uint> {
        std::vector<uint> idx(count);
#pragma omp parallel for schedule(static, 4096)
        for (long long i = 0; i < (long long)count; ++i) idx[i] = start + (uint)i;

        psort(idx.begin(), idx.end(), [&, is_left](uint a, uint b) {
            uint da = is_left ? new_degrees[a] - 2 : new_degrees[a] - 1;
            uint db = is_left ? new_degrees[b] - 2 : new_degrees[b] - 1;
            uint ba = degree_bucket(da), bb = degree_bucket(db);
            if (ba != bb) return ba < bb;
            if (da != db) return da < db;
            return a < b;
        });

        return idx;
    };

    std::vector<uint> left_nodes  = sort_bucketed(0, nLeft, true);
    std::vector<uint> right_nodes = sort_bucketed(nLeft, nRight, false);

    ParallelArray<uint> global_map(N);
    global_map.first_touch();

#pragma omp parallel for
    for (long long i = 0; i < (long long)nLeft; ++i) global_map[left_nodes[i]] = (uint)i;
#pragma omp parallel for
    for (long long i = 0; i < (long long)nRight; ++i) global_map[right_nodes[i]] = nLeft + (uint)i;

    std::vector<uint>().swap(left_nodes);
    std::vector<uint>().swap(right_nodes);

    std::vector<uint> final_deg(N, 0);
#pragma omp parallel for
    for (long long old_id = 0; old_id < (long long)N; ++old_id) {
        final_deg[global_map[old_id]] = new_degrees[old_id];
    }

    std::vector<uint> final_offset(N + 1);
    parallel_exclusive_scan(final_deg.data(), final_offset.data(), (size_t)N);
    uint final_total = final_offset[N];

    ParallelArray<uint> final_neighbors(final_total);
    final_neighbors.first_touch();

#pragma omp parallel for schedule(dynamic, 256)
    for (long long old_u = 0; old_u < (long long)N; ++old_u) {
        uint new_u = global_map[old_u];
        uint s_off = offset[old_u];
        uint d_off = final_offset[new_u];
        uint deg   = new_degrees[old_u];

        for (uint k = 0; k < deg; ++k) {
            if (k + 16 < deg) {
                PFETCH(&global_map[neighbors[s_off + k + 16]]);
            }
            final_neighbors[d_off + k] = global_map[neighbors[s_off + k]];
        }

        uint* b = final_neighbors.data() + d_off;
        uint* e = b + deg;
        if (deg <= 32) insertion_sort_u32(b, e);
        else std::sort(b, e);
    }

    neighbors.reset();
    std::vector<uint>().swap(offset);
    std::vector<uint>().swap(new_degrees);
    global_map.reset();

    uint nEdge = final_total;
    uint bipartite_node = nLeft;

    auto write_binary = [&](const std::string& path) {
        const size_t header_size = 12;
        size_t total_size = header_size + 8ull * (size_t)N + 4ull * (size_t)nEdge;

        MMapWriter writer(path, total_size);
        char* base = writer.data();

        std::memcpy(base + 0, &nEdge,          4);
        std::memcpy(base + 4, &N,              4);
        std::memcpy(base + 8, &bipartite_node, 4);

#pragma omp parallel for schedule(dynamic, 256)
        for (long long i = 0; i < (long long)N; ++i) {
            size_t boff_i = 8ull * (size_t)i + 4ull * (size_t)final_offset[i];
            char* p = base + header_size + boff_i;
            uint id = (uint)i;
            uint deg = final_deg[i];
            std::memcpy(p,     &id,  4);
            std::memcpy(p + 4, &deg, 4);
            if (deg > 0)
                std::memcpy(p + 8, final_neighbors.data() + final_offset[i], 4ull * deg);
        }
    };

    write_binary(outfile);

    std::cout << "Nodes:   " << N      << "\n"
              << "2|E|  :  " << nEdge  << "\n"
              << "nLeft :  " << nLeft  << "\n"
              << "nRight:  " << nRight << std::endl;
}

int main(int argc, char* argv[]) {
    omp_set_dynamic(0);

    CliArgs args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return 1;
    }
    if (args.help) {
        print_usage(argv[0]);
        return 0;
    }
    if (args.infile.empty() || args.outfile.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        Convert(args.infile, args.outfile);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}