#include <Util.h>
#include "args.hxx"
#include "iTraversal.h"
#include <cstring>
#include <recorder.h>
#include <stddef.h>
#include <sys/types.h>

#define FILELEN 1024
int main(int argc, char** argv) {
	char filepath[1024] = ".........";

	int k = 1;
	int num = 1000;
	int isquiete = 0;

	args::ArgumentParser parser(
		"iTraversal, an algorithm for enumerating all maximal biplexes\n");

	args::HelpFlag help(parser, "help", "Display this help menu", { 'h', "help" });
	args::Group required(parser, "", args::Group::Validators::All);

	args::ValueFlag<std::string> benchmark_file(
		parser, "benchmark", "Path to benchmark", { 'f', "file" }, "");

	args::ValueFlag<int> K(parser, "para k", "The parameter k", { 'k', "k" }, 1);

	args::ValueFlag<int> Results(parser, "Num of results", "Num of results",
		{ 'r', "r" }, 1000);

	args::ValueFlag<int> Quiete(parser, "quiete", "quiete or not", { 'q', "q" }, 0);

	try {
		parser.ParseCLI(argc, argv);
	}
	catch (args::Help) {
		std::cout << parser;
		return 0;
	} catch (args::ParseError e) {
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		return 0;
	} catch (args::ValidationError e) {
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		return 0;
	}

	strncpy(filepath, args::get(benchmark_file).c_str(), FILELEN);
	k = args::get(K);
	isquiete = args::get(Quiete);
	num = args::get(Results);

	if (k < 0 || num < 0) {
		fprintf(stderr, "k, theta and num should be at least 0\n");
		exit(-1);
	}

	if (isquiete > 0)
		YES::OutputResults2 = 1;

	int bi = 0;
	vector<int> degree;
	vector<vector<int> > Graph;
	auto [sucess, graph_size, edge_num] = ReadBinGraph(filepath, Graph, degree, bi);
	fmt::println("Graph size = {}, bipartite index = {}, edge num = {}", graph_size, bi, edge_num);
	for (uint i = 0;; i++) {
		int tmp = pow(10, i);
		if (tmp < num)
			checkpoints.insert(tmp);
		else {
			checkpoints.insert(num);
			break;
		}
	}
	cpu::recorder.init(filepath, k, num, checkpoints, ".cpu_sota");
	std::filesystem::path benchmark_file_path = filepath;
	if (num == 1000000) {
		count_logger.init("cpu", benchmark_file_path.string(), k);
		count_logger.count(0);
	}

	YES::iTraversal iminer(Graph, degree, graph_size, bi, k, num);
	iminer.miner();
	cout << fixed << setprecision(5);
	std::cout << "\n===== Performance Report =====" << std::endl;
	std::cout << "Result Count (r)\tTime (ms)" << std::endl;
	for (const auto& record : iminer.time_records) {
		std::cout << record.first << "\t\t" << record.second << std::endl;
	}
	std::cout << "==============================" << std::endl;
	return 0;
}
