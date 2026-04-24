#ifndef SGMOD22_MBPE_ITRAVERSAL2_H
#define SGMOD22_MBPE_ITRAVERSAL2_H

#include <iostream>
#include <vector>
#include <string>
#include "cpp-btree-1.0.1/btree_set.h"
#include "Util.h"
#include "cpp-btree-1.0.1/btree.h"
#include <cstring>
#include <fmt/core.h>
#include <algorithm>
#include <recorder.h>

using namespace std;

namespace YES {
    int OutputResults2 = 0;

    class iTraversal {
    public:
        iTraversal(int **Graph, int *degree, int
        Graph_size, int
                   Bipartite_index, int
                   Epsilon, int
                   total_num);

        iTraversal(vector <vector<int>>
                   Graph, vector<int>
                   degree, int
                   Graph_size, int
                   Bipartite_index, int
                   Epsilon, int
                   total_num);

        void
        miner();

        void
        ExtendToMax(vector<int> &X, vector<int> &Y);

        void
        Enumerate(vector<int> &X, vector<int> &Y, vector<int> &Full_X, vector<int> &FIX_X, int index,
                  int output);

        void
        TransToString(vector<int> &X, vector<int> &Y, vector<int> &FIX, string &res);

        void
        TransToString2(vector<int> &X, vector<int> &Y, string &res);

        inline bool
        FindNext(vector<int> &X, vector<int>::iterator arr[], int S);

        inline bool
        FindNext(vector<int> &X, vector <vector<int>::iterator> &arr, int sizeofArr);

        bool
        checkY(vector<int> &X, int yuj, vector<int> &cons_Y, vector<int>::iterator Inc[], vector<int> &FIX_X);

        bool
        checkY(vector<int> &X, int emplaceIncCount, vector<int> &pivotNeighbor, vector <vector<int>::iterator> Inc,
               vector<int> &FIX_X);

        void
        GetSAT();

        int total_num;
        int edges = 0;
        int res_count = 0;
        TimePoint start_time;
        vector <pair<int, double>> time_records;

    private:

        int **Graph;
        vector <vector<int>> _Graph;
        int Graph_size;
        int Bipartite_index;
        int *_degree;
        vector<int> degree;
        int Epsilon;
        string *str_list;
        char temp_char[15];
        int *G_index;
        int *G_degree;
        int *G_temp;
        int *G_temp2;
        int *G_temp3;
        int *G_temp4;
        int *G_mark;
        int *G_exc;
        int temp_yu;
        btree::btree_set <string> Btree;
        vector<int> X_a, Y_a;
        vector<int>::iterator temp_iter, temp_iter1, temp_iter2, temp_iter3;
        bool temp_check, temp_check1, temp_check2;
        int temp_i, temp_j, temp_v1, temp_v2, temp_count1, temp_count2, temp_node1, temp_node2, ac_node;
    };

    iTraversal::iTraversal(int **Graph, int *degree, int Graph_size, int Bipartite_index, int Epsilon, int total_num) {
        this->start_time = Clock::now();
        this->Graph = Graph;
        this->_degree = degree;
        this->Graph_size = Graph_size;
        this->Bipartite_index = Bipartite_index;
        this->Epsilon = Epsilon;
        this->total_num = total_num;

        G_index = new int[Graph_size];
        G_degree = new int[Graph_size];
        G_temp = new int[Graph_size];
        G_temp2 = new int[Graph_size];
        G_temp3 = new int[Graph_size];
        G_mark = new int[Graph_size];
        G_exc = new int[Graph_size];
        str_list = new string[Graph_size];
        G_temp4 = new int[Graph_size];

        bool inital = true;

        for (int i = 0; i < Graph_size; ++i) {
            G_index[i] = 0;
            G_degree[i] = 0;
            G_temp[i] = 0;
            G_temp2[i] = 0;
            G_temp3[i] = 0;
            G_mark[i] = 0;
            G_exc[i] = 0;
            G_temp4[i] = 0;
            str_list[i] = string(itoa(i, temp_char, 10)) + "&";
            if (i < Bipartite_index && degree[i] > (Graph_size - Bipartite_index - Epsilon)) {
                inital = false;
            }
        }

        if (inital) {
            res_count++;
            if (res_count % 1000 == 0)
                count_logger.count(res_count);
            if (checkpoints.count(res_count)) {
                TimePoint current_time = Clock::now();

                std::chrono::duration<double, std::milli> elapsed = current_time - start_time;
                double ms = elapsed.count();

                time_records.push_back({res_count, ms});
                cpu::recorder.record(to_string(res_count), ms);
                fmt::println("Output {}, cost Time: {} ms", res_count, ms);
            }

        }
    }

    iTraversal::iTraversal(vector <vector<int>> Graph,
                           vector<int> degree,
                           int Graph_size,
                           int Bipartite_index,
                           int Epsilon,
                           int total_num) {
        this->start_time = Clock::now();
        this->_Graph = Graph;
        this->degree = degree;
        this->Graph_size = Graph_size;
        this->Bipartite_index = Bipartite_index;
        this->Epsilon = Epsilon;
        this->total_num = total_num;

        G_index = new int[Graph_size];
        G_degree = new int[Graph_size];
        G_temp = new int[Graph_size];
        G_temp2 = new int[Graph_size];
        G_temp3 = new int[Graph_size];
        G_mark = new int[Graph_size];
        G_exc = new int[Graph_size];
        str_list = new string[Graph_size];
        G_temp4 = new int[Graph_size];

        bool inital = true;

        for (int i = 0; i < Graph_size; ++i) {
            G_index[i] = 0;
            G_degree[i] = 0;
            G_temp[i] = 0;
            G_temp2[i] = 0;
            G_temp3[i] = 0;
            G_mark[i] = 0;
            G_exc[i] = 0;
            G_temp4[i] = 0;
            str_list[i] = string(itoa(i, temp_char, 10)) + "&";
            if (i < Bipartite_index && degree[i] > (Graph_size - Bipartite_index - Epsilon)) {
                inital = false;
            }
        }

        if (inital) {
            res_count++;
            if (res_count % 1000 == 0)
                count_logger.count(res_count);
            if (checkpoints.count(res_count)) {
                TimePoint current_time = Clock::now();

                std::chrono::duration<double, std::milli> elapsed = current_time - start_time;
                double ms = elapsed.count();

                time_records.push_back({res_count, ms});
                cpu::recorder.record(to_string(res_count), ms);
                fmt::println("Output {}, cost Time: {} ms", res_count, ms);
            }

        }
    }

    inline bool
    iTraversal::FindNext(vector<int> &X, vector<int>::iterator arr[], int S) {
        temp_iter = X.end();
        for (temp_i = S - 1; temp_i >= 0; --temp_i) {
            ++arr[temp_i];
            if (arr[temp_i] != temp_iter) {
                temp_check = true;
                for (temp_j = temp_i + 1; temp_j < S; ++temp_j) {
                    arr[temp_j] = arr[temp_j - 1];
                    ++arr[temp_j];
                    if (arr[temp_j] == temp_iter) {
                        temp_check = false;
                        break;
                    }
                }
                if (temp_check)
                    return temp_check;
            }
        }
        return false;
    }


    inline bool
    iTraversal::FindNext(vector<int> &X, vector <vector<int>::iterator> &arr, int sizeofArr) {
        /*!
         * @brief 每次调用都会更改一次arr中，以遍历在X中的每种可能
         * @param X: 排列组合的原数组
         * @param sizeofArr: 排列组合的结果
         * @param S: arr的大小
         * @return 是否成功，如果发现现在arr里的是最后一种可能了，无法再遍历了就返回false
         * */
        for (temp_i = sizeofArr - 1; temp_i >= 0; --temp_i) {
            ++arr[temp_i];
            if (arr[temp_i] != X.end()) {
                temp_check = true;
                for (temp_j = temp_i + 1; temp_j < sizeofArr; ++temp_j) {
                    arr[temp_j] = arr[temp_j - 1];
                    ++arr[temp_j];
                    if (arr[temp_j] == X.end()) {
                        temp_check = false;
                        break;
                    }
                }
                if (temp_check)
                    return temp_check;
            }
        }
        return false;
    }

    void
    iTraversal::miner() {
        int node = 0;
        bool extension = true;
        vector<int>::iterator iter;
        vector<int>::iterator iter1;
        bool ext = false;
        int temp_node;
        int output = 0;

        vector<int> X;
        vector<int> Y;
        vector<int> FIX_X;
        vector<int> FIX_Y;
        vector<int> pivotDidnotAccessNodes, new_X, new_FIX_X, _pivotDidnotAccessNode_Inc;
        /*
     * x代表左边，y代表右边
     * */
        vector<int>::iterator pivotDidnotAccessNode_Inc[Epsilon];
        bool redo = true;
        bool exc_yu = true;

        for (int pivot = 0; pivot < Bipartite_index; ++pivot) {
            if (res_count >= total_num)
                return;

            X.clear();
            Y.clear();
            FIX_X.clear();
            FIX_Y.clear();

            X.push_back(pivot);
            G_index[pivot] = 1;

            for (auto neighbor_node: _Graph[pivot]) {
                G_index[neighbor_node] = 1;
                G_degree[neighbor_node] = 1;
                FIX_Y.push_back(neighbor_node);
                Y.push_back(neighbor_node);
                G_mark[neighbor_node] = 1;

                for (auto twoHopNeighborNode: _Graph[neighbor_node])
                    G_degree[twoHopNeighborNode]++;
            }

            pivotDidnotAccessNodes.clear();
            for (int rightNodes = Bipartite_index; rightNodes < Graph_size; ++rightNodes)
                if (G_index[rightNodes] == 0)
                    pivotDidnotAccessNodes.push_back(rightNodes);

            int s = min((int) pivotDidnotAccessNodes.size(), Epsilon);
            temp_i = 0;
            for (auto it = pivotDidnotAccessNodes.begin(); temp_i < s; ++it, ++temp_i)
                pivotDidnotAccessNode_Inc[temp_i] = it;
            _pivotDidnotAccessNode_Inc.assign(pivotDidnotAccessNodes.begin(), pivotDidnotAccessNodes.begin() + s);



            redo = true;
            while (redo) {
                for (auto node: _pivotDidnotAccessNode_Inc) {
                    G_index[node] = 1;
                    for (auto neighborOfNode: _Graph[node])
                        G_degree[neighborOfNode]++;
                }
                extension = true;
                exc_yu = true;
                for (auto node: _pivotDidnotAccessNode_Inc) {
                    for (auto neighborOfNode: _Graph[node]) {
                        if (G_index[neighborOfNode] == 0 && G_degree[neighborOfNode] == degree[pivot] + s
                            && pivot != neighborOfNode) {
                            /*
                         * 此时node是右边的顶点，非pivot邻居，且ID属于[u, u+s]的点
                         * 此时neighborOfNode是左边的顶点
                         * 条件1： 此时左边顶点都是0
                         * 条件2： 扩展左边顶点的条件，比如此时pivot的度是3，而k = 2，我需要制造pivot有k=2个不连接的顶点的子图
                         * 那么 degree = k + degree[pivot]刚好可以构造这样的顶点
                         * 【不可能在第一个pivot满足这个条件，因为此时由G_degree描述的子图在右边只包含了pivot的邻居】
                         * */
                            if (neighborOfNode < pivot) {
                                extension = false;
                                break;
                            }
                            FIX_X.push_back(neighborOfNode);
                            G_index[neighborOfNode] = 2;
                        }
                    }
                }

                if (extension) {
                    for (auto node: _pivotDidnotAccessNode_Inc) {
                        Y.emplace_back(node);
                        FIX_Y.emplace_back(node);
                        G_mark[node] = 1;
                    }

                    if (Y.size() + s > Epsilon) {
                        for (auto node: _pivotDidnotAccessNode_Inc) {
                            for (auto neighborsOfNode: _Graph[node]) {
                                if (G_index[neighborsOfNode] == 0 && G_degree[neighborsOfNode] + Epsilon >= Y.size()) {
                                    /*
                                 * 如果这个左边顶点没有被访问过, 并且在目前由X（左），Y（右），Y邻居（左）组成的子图中
                                 * neighborsOfNode（左）还连接着的顶点(右边）个数 + Epsilon >= Y.size()。【y.size()就是右边的顶点个数】
                                 * 在这幅子图中, 这说明在这幅子图中，在左边加入neighborsOfNode，不会破坏k-biplex的定义
                                 * */
                                    ext = true;
                                    for (auto neighborsOf_neighborsOfNode: _Graph[neighborsOfNode])
                                        G_temp4[neighborsOf_neighborsOfNode] = 1;
                                    /*
                                 * neighborsOfNode的邻居都打上标签
                                 * */
                                    for (auto y: Y) {
                                        if (G_temp4[y] == 0 && (int) (X.size()) + 1 - G_degree[y] > Epsilon) {
                                            /* 这个操作主要避免在左边加入新的顶点时导致右边顶点违背biplex的定义
                                         * 在Y里面, 选那些不是neighborsOfNode(左)邻居(右)的顶点，称其为y，这些顶点情况决定了此时的neighborsOfNode
                                         * 是否可以被加入到左边
                                         * (int)(X.size()) + 1指的是假设neighborsOfNode被加入，而(int)(X.size()) + 1 - G_degree[y]，指的是
                                         * 对于y来说，当neighborsOfNode之后，其未连接的顶点的数量
                                         * 判断的重要依据就是y未连接的顶点的数量是否大于Epsilon
                                         * */
                                            ext = false;
                                            break;
                                        }
                                    }
                                    for (auto neighborsOf_neighborsOfNode: _Graph[neighborsOfNode])
                                        G_temp4[neighborsOf_neighborsOfNode] = 0;

                                    if (ext) {
                                        if (neighborsOfNode < pivot) {
                                            exc_yu = false;
                                            break;
                                        }

                                        X.emplace_back(neighborsOfNode);
                                        G_index[neighborsOfNode] = 1;
                                        for (auto neighborsOf_neighborsOfNode: _Graph[neighborsOfNode])
                                            G_degree[neighborsOf_neighborsOfNode]++;
                                    }
                                }
                            }
                        }

                        if (degree[pivot] != 0 && exc_yu) {
                            int temp_node = _Graph[pivot][0];
                            for (auto neighborsOfTempNode: _Graph[temp_node]) {
                                if (G_index[neighborsOfTempNode] == 0
                                    && G_degree[neighborsOfTempNode] >= Y.size() - Epsilon) {
                                    ext = true;
                                    for (auto neighborsOf_neighborsOfTempNode: _Graph[neighborsOfTempNode])
                                        G_temp4[neighborsOf_neighborsOfTempNode] = 1;

                                    for (auto y: Y) {
                                        if (!G_temp4[y] && G_degree[y] < X.size() - Epsilon + 1) {
                                            ext = false;
                                            break;
                                        }
                                    }

                                    for (auto neighborsOf_neighborsOfTempNode: _Graph[neighborsOfTempNode])
                                        G_temp4[neighborsOf_neighborsOfTempNode] = 0;

                                    if (ext) {
                                        if (neighborsOfTempNode < pivot) {
                                            exc_yu = false;
                                            break;
                                        }
                                        X.push_back(neighborsOfTempNode);
                                        G_index[neighborsOfTempNode] = 1;

                                        for (auto neighborsOfNode: _Graph[neighborsOfTempNode])
                                            G_degree[neighborsOfNode]++;

                                    }
                                }
                            }
                        }
                    } else {
                        for (int leftNode = 0; leftNode < Bipartite_index; ++leftNode) {
                            if (G_index[leftNode] == 0 && G_degree[leftNode] >= (int) Y.size() - Epsilon) {
                                ext = true;
                                for (auto neighborsOfLeftNode: _Graph[leftNode])
                                    G_temp4[neighborsOfLeftNode] = 1;
                                for (auto y: Y) {
                                    if (!G_temp4[y] && G_degree[y] < (int) (X.size()) - Epsilon + 1) {
                                        ext = false;
                                        break;
                                    }
                                }
                                for (auto node: _Graph[leftNode])
                                    G_temp4[node] = 0;

                                if (ext) {
                                    if (leftNode < pivot) {
                                        exc_yu = false;
                                        break;
                                    }
                                    X.push_back(leftNode);
                                    G_index[leftNode] = 1;
                                    for (auto neighborsOfLeftNode: _Graph[leftNode])
                                        G_degree[neighborsOfLeftNode]++;

                                }
                            }
                        }
                    }

                    if (exc_yu) {
                        string temp_res;
                        TransToString(X, Y, FIX_X, temp_res);
                        if (Btree.find(temp_res) == Btree.end()) {
                            if (output % 2 == 0) {
                                res_count++;
                                if (res_count % 1000 == 0)
                                    count_logger.count(res_count);
                                if (checkpoints.count(res_count)) {
                                    TimePoint current_time = Clock::now();

                                    std::chrono::duration<double, std::milli> elapsed = current_time - start_time;

                                    double ms = elapsed.count();
                                    time_records.push_back({res_count, ms});
                                    cpu::recorder.record(to_string(res_count), ms);
                                    fmt::println("Output {}, cost Time: {} ms", res_count, ms);
                                }
                                if (OutputResults2)
                                    cout << temp_res << endl;

                                if (res_count >= total_num)
                                    return;
                            }
                            Btree.insert(temp_res);
                            new_X.clear();
                            new_FIX_X.clear();
                            new_X.assign(X.begin(), X.end());
                            new_FIX_X.assign(FIX_X.begin(), FIX_X.end());

                            Enumerate(new_X, Y, FIX_Y, new_FIX_X, 2, 0);
                            if (res_count >= total_num)
                                return;

                            if (output % 2) {
                                res_count++;
                                if (res_count % 1000 == 0)
                                    count_logger.count(res_count);
                                if (checkpoints.count(res_count)) {
                                    TimePoint current_time = Clock::now();

                                    std::chrono::duration<double, std::milli> elapsed = current_time - start_time;

                                    double ms = elapsed.count();
                                    time_records.push_back({res_count, ms});
                                    cpu::recorder.record(to_string(res_count), ms);
                                    fmt::println("Output {}, cost Time: {} ms", res_count, ms);
                                }
                                if (OutputResults2)
                                    cout << temp_res << endl;

                                if (res_count >= total_num)
                                    return;
                            }
                        }
                    }

                    for (auto it = X.begin() + 1; it != X.end(); it++) {
                        for (auto node: _Graph[*it])
                            G_degree[node]--;
                        G_index[*it] = 0;
                    }

                    for (auto node: FIX_X)
                        G_index[node] = 0;
                    FIX_X.clear();
                    for (auto node: _pivotDidnotAccessNode_Inc) {
                        G_mark[node] = 0;
                        FIX_Y.pop_back();
                    }
                    X.clear();
                    X.push_back(pivot);

                    for (temp_i = 0; temp_i < s; ++temp_i) {
                        G_index[Y[Y.size() - 1]] = 0;
                        Y.pop_back();
                    }
                }
                for (auto node: _pivotDidnotAccessNode_Inc) {
                    G_index[node] = 0;
                    for (auto neighborsOfNode: _Graph[node])
                        G_degree[neighborsOfNode]--;
                }
                for (auto node: FIX_X)
                    G_index[node] = 0;
                FIX_X.clear();
                redo = FindNext(pivotDidnotAccessNodes, pivotDidnotAccessNode_Inc, s);
                _pivotDidnotAccessNode_Inc.clear();
                if (not redo)break;
                for (auto it: pivotDidnotAccessNode_Inc)
                    _pivotDidnotAccessNode_Inc.emplace_back(*it);
            }

            for (auto x: X)
                G_index[x] = 0;
            for (auto node: _Graph[pivot]) {
                G_index[node] = 0;
                G_degree[node] = 0;
                G_mark[node] = 0;
                for (auto neighborsOfNode: _Graph[node])
                    G_degree[neighborsOfNode] = 0;
            }
            G_exc[pivot] = 1;
        }
    }

    void
    iTraversal::GetSAT() {
        cout << "----------------- Satistical Infomation -----------------" << endl;
        cout << "total number: " << res_count << endl;
        cout << "edges: " << edges << endl;
        cout << "---------------------------------------------------------" << endl;
    }

    void
    iTraversal::TransToString(vector<int> &X, vector<int> &Y, vector<int> &FIX, string &res) {
        ++edges;

        X_a.clear();
        X_a.assign(X.begin(), X.end());
        X_a.insert(X_a.end(), FIX.begin(), FIX.end());

        Y_a.clear();
        Y_a.assign(Y.begin(), Y.end());

        res.reserve(2 * (X.size() + Y.size()));

        sort(X_a.begin(), X_a.end());
        for (auto x: X_a)
            res += str_list[x];

        sort(Y_a.begin(), Y_a.end());
        for (auto y: Y_a)
            res += str_list[y];
    }

    void
    iTraversal::TransToString2(vector<int> &X, vector<int> &Y, string &res) {
        ++edges;
        res.reserve(2 * (X.size() + Y.size()));

        temp_iter1 = X.begin();
        temp_iter2 = X.end();
        sort(temp_iter1, temp_iter2);
        for (; temp_iter1 != temp_iter2; ++temp_iter1) {
            res += str_list[*temp_iter1];
        }

        temp_iter1 = Y.begin();
        temp_iter2 = Y.end();
        sort(temp_iter1, temp_iter2);
        for (; temp_iter1 != temp_iter2; ++temp_iter1) {
            res += str_list[*temp_iter1];
        }
    }

    void
    iTraversal::Enumerate(vector<int> &X, vector<int> &Y, vector<int> &Full_Y, vector<int> &FIX_X, int depth,
                          int output) {
        if (Y.size() == 0 || res_count >= total_num)
            return;

        output = 0;
        int count_yu = 0;
        bool check_miss = true;
        bool extension = true;
        bool ext = false;
        bool yu_exc = true;
        vector <vector<int>::iterator> Inc, FIX;
        Inc.reserve(Epsilon);
        FIX.reserve(Epsilon);
        int s, NB_S;
        int notPivotPlexCount = 0, notPivotNotPlexCount = 0;
        bool redo = true, redo2;
        vector<int> F_Y, new_FIX_X, new_new_X, pivotNeighbors, notPivotNeighbors_Renum, new_X, temp_FIX_X, NB;

        for (int pivot = 0; pivot < Bipartite_index; ++pivot) {
            if (res_count >= total_num)
                return;

            if (G_index[pivot] > 0 or G_exc[pivot] > 0)
                continue;
            /*
             * pivot不是前面被执行过的左边顶点G_exc。
             * pivot不是目前子图中的左边顶点
             * */


            pivotNeighbors.clear();
            notPivotNeighbors_Renum.clear();
            check_miss = true;
            extension = true;

            G_index[pivot] = 1;
            for (auto neighborOfPivot: _Graph[pivot]) {
                G_temp[neighborOfPivot] = 1;
                G_degree[neighborOfPivot]++;
                /*
                 * 将pivot的邻居压入子图
                 * */
            }
            /*
             * 此时得到的就是(L∪{v}, R), 几乎完美子图
             * */
            for (auto y: Y) {
                /*
                 * 在子图里，右边不是pivot的顶点，被剔除出子图
                 * 我们把与v（也就是这里的pivot)链接的顶点设置为R_keep, 其他为R_enum
                 * 因为所有的局部解都包含R_keep，所以不需要被遍历，而只需要在下面遍历R_enum,
                 * */
                if (G_temp[y] == 0) {
                    notPivotNeighbors_Renum.emplace_back(y);
                    G_index[y] = 0;
                    for (auto neighborOfY: _Graph[y])
                        G_degree[neighborOfY]--;
                } else
                    pivotNeighbors.emplace_back(y);
            }
            for (auto neighborOfPivot: _Graph[pivot])
                G_temp[neighborOfPivot] = 0;
            for (auto notNeighbor: notPivotNeighbors_Renum) {
                /*
                 * 如果载入的顶点邻居和子图右边无交集
                 * 或者左边顶点满足k-1 biplex的定义(不知道为什么是这个)
                 * */
                if (X.size() - G_degree[notNeighbor] <= Epsilon - 1 or pivotNeighbors.empty()) {
                    check_miss = false;
                    break;
                }
            }
            yu_exc = true;
            if (check_miss) {
                extension = true;
                temp_FIX_X.clear();
                for (auto neighborOfPivot0: _Graph[pivotNeighbors[0]]) {
                    if (G_index[neighborOfPivot0] == 0 and G_degree[neighborOfPivot0] == pivotNeighbors.size() and
                        neighborOfPivot0 != pivot) {
                        if (G_exc[neighborOfPivot0] > 0) {
                            extension = false;
                            break;
                        }
                        temp_FIX_X.push_back(neighborOfPivot0);
                        G_index[neighborOfPivot0] = 2;
                    }
                }

                if (extension) {
                    temp_v1 = (int) X.size() - Epsilon + 1;
                    temp_v2 = (int) pivotNeighbors.size() - Epsilon + 1;
                    if (X.size() > Epsilon) {
                        temp_count1 = 0;
                        for (auto x: X) {
                            if (temp_count1 > Epsilon)break;
                            temp_count1++;
                            for (auto neighborOfX: _Graph[x]) {
                                if (G_index[neighborOfX] == 0 && G_degree[neighborOfX] >= temp_v1) {
                                    extension = false;
                                    for (auto twoHopNeighborOfX: _Graph[neighborOfX])
                                        G_temp4[twoHopNeighborOfX] = 1;

                                    for (auto _x: X) {
                                        if (!G_temp4[_x] && G_degree[_x] < temp_v2) {
                                            extension = true;
                                            break;
                                        }
                                    }

                                    if (G_temp3[pivot] == 0 && G_degree[pivot] < temp_v2)
                                        extension = true;
                                    if (!extension && not FIX_X.empty()) {
                                        temp_count2 = 0;
                                        for (auto fix_x: FIX_X)
                                            if (!G_temp4[fix_x])
                                                ++temp_count2;
                                        if (G_degree[neighborOfX] < temp_v1 + temp_count2)
                                            extension = true;
                                    }
                                    for (auto twoHopNeighborOfX: _Graph[neighborOfX])
                                        G_temp4[twoHopNeighborOfX] = 0;
                                    if (!extension)
                                        break;
                                }
                            }
                            if (!extension)
                                break;
                        }
                    } else {
                        for (int leftNodes = Bipartite_index; leftNodes < Graph_size; ++leftNodes) {
                            if (G_index[leftNodes] == 0 && G_degree[leftNodes] >= temp_v1) {
                                extension = false;
                                for (auto neighborOfLeftNode: _Graph[leftNodes])
                                    G_temp4[neighborOfLeftNode] = 1;

                                for (auto x: X) {
                                    if (!G_temp4[x] && G_degree[x] < temp_v2) {
                                        extension = true;
                                        break;
                                    }
                                }

                                if (G_temp4[pivot] == 0 && G_degree[pivot] < temp_v2)
                                    extension = true;

                                if (!extension && not FIX_X.empty()) {
                                    temp_count2 = 0;
                                    for (auto fix_x: FIX_X)
                                        if (not G_temp4[fix_x])
                                            ++temp_count2;
                                    if (G_degree[leftNodes] < temp_v1 + temp_count2)
                                        extension = true;
                                }
                                for (auto neighborOfLeftNode: _Graph[leftNodes])
                                    G_temp4[neighborOfLeftNode] = 0;
                                if (!extension)
                                    break;
                            }
                        }
                    }
                }

                if (extension) {
                    new_X.clear();
                    new_X.push_back(pivot);
                    temp_v1 = (int) X.size() + 2 - Epsilon;
                    temp_v2 = (int) pivotNeighbors.size() - Epsilon;

                    if (pivotNeighbors.size() > Epsilon) {
                        temp_count1 = 0;
                        for (auto pivotNeighbor: pivotNeighbors) {
                            if (temp_count1 > Epsilon)break;
                            temp_count1++;
                            for (auto twoHopNeighborOfPivot: _Graph[pivotNeighbor]) {
                                if (twoHopNeighborOfPivot != pivot && G_index[twoHopNeighborOfPivot] == 0 &&
                                    G_degree[twoHopNeighborOfPivot] >= temp_v2) {
                                    temp_check = true;
                                    for (auto threeHopNeighborOfPivot: _Graph[twoHopNeighborOfPivot])
                                        G_temp4[threeHopNeighborOfPivot] = 1;
                                    for (auto _pivotNeighbor: pivotNeighbors) {
                                        if (!G_temp4[_pivotNeighbor] && G_degree[_pivotNeighbor] < temp_v1) {
                                            temp_check = false;
                                            break;
                                        }
                                    }
                                    for (auto threeHopNeighborOfPivot: _Graph[twoHopNeighborOfPivot])
                                        G_temp4[threeHopNeighborOfPivot] = 0;
                                    if (temp_check) {
                                        if (G_exc[twoHopNeighborOfPivot]) {
                                            yu_exc = false;
                                            break;
                                        }
                                        new_X.emplace_back(twoHopNeighborOfPivot);
                                        ++temp_v1;
                                        G_index[twoHopNeighborOfPivot] = 1;
                                        for (auto threeHopNeighborOfPivot: _Graph[twoHopNeighborOfPivot])
                                            G_degree[threeHopNeighborOfPivot]++;
                                    }
                                }
                            }
                            if (G_degree[pivotNeighbor] == temp_v1 - 1)
                                break;
                            if (!yu_exc)
                                break;
                        }
                    } else {
                        temp_check1 = true;
                        for (auto pivotNeighbor: pivotNeighbors) {
                            if (G_degree[pivotNeighbor] == temp_v1 - 1) {
                                for (auto twoHopNeighborOfPivot: _Graph[pivotNeighbor]) {
                                    if (twoHopNeighborOfPivot != pivot && G_index[twoHopNeighborOfPivot] == 0 &&
                                        G_degree[twoHopNeighborOfPivot] >= temp_v2) {
                                        temp_check = true;
                                        for (auto threeHopNeighborOfPivot: _Graph[twoHopNeighborOfPivot])
                                            G_temp4[threeHopNeighborOfPivot] = 1;

                                        for (auto _pivotNeighbor: pivotNeighbors) {
                                            if (!G_temp4[_pivotNeighbor] && G_degree[_pivotNeighbor] < temp_v1) {
                                                temp_check = false;
                                                break;
                                            }
                                        }
                                        for (auto threeHopNeighborOfPivot: _Graph[twoHopNeighborOfPivot])
                                            G_temp4[threeHopNeighborOfPivot] = 0;
                                        if (temp_check) {
                                            if (G_exc[twoHopNeighborOfPivot]) {
                                                yu_exc = false;
                                                break;
                                            }
                                            new_X.push_back(twoHopNeighborOfPivot);
                                            ++temp_v1;
                                            G_index[twoHopNeighborOfPivot] = 1;
                                            for (auto threeHopNeighborOfPivot: _Graph[twoHopNeighborOfPivot])
                                                G_degree[threeHopNeighborOfPivot]++;
                                        }
                                    }
                                }
                                break;
                            }
                        }
                        if (!yu_exc)
                            break;

                        for (int leftNodes = 0; leftNodes < Bipartite_index && temp_check1; ++leftNodes) {
                            if (leftNodes != pivot && G_index[leftNodes] == 0 && G_degree[leftNodes] >= temp_v2) {
                                temp_check = true;
                                for (auto neighborOfLeftNode: _Graph[leftNodes])
                                    G_temp4[neighborOfLeftNode] = 1;
                                for (auto pivotNeighbor: pivotNeighbors) {
                                    if (!G_temp4[pivotNeighbor] && G_degree[pivotNeighbor] < temp_v1) {
                                        temp_check = false;
                                        break;
                                    }
                                }
                                for (auto neighborOfLeftNode: _Graph[leftNodes])
                                    G_temp4[neighborOfLeftNode] = 0;
                                if (temp_check) {
                                    if (G_exc[leftNodes]) {
                                        yu_exc = false;
                                        break;
                                    }
                                    new_X.push_back(leftNodes);
                                    G_index[leftNodes] = 1;
                                    ++temp_v1;
                                    for (auto neighborOfLeftNode: _Graph[leftNodes])
                                        G_degree[neighborOfLeftNode]++;
                                }
                            }
                        }
                    }
                    if (yu_exc) {

                        X_a.clear();
                        F_Y.clear();
                        X_a.insert(X_a.end(), X.begin(), X.end());
                        X_a.insert(X_a.end(), new_X.begin(), new_X.end());
                        X_a.insert(X_a.end(), FIX_X.begin(), FIX_X.end());
                        X_a.insert(X_a.end(), temp_FIX_X.begin(), temp_FIX_X.end());
                        F_Y.insert(F_Y.end(), pivotNeighbors.begin(), pivotNeighbors.end());
                        string temp_res;
                        TransToString2(X_a, F_Y, temp_res);
                        if (Btree.find(temp_res) == Btree.end()) {
                            if (output % 2 == 0) {
                                ++res_count;
                                if (res_count % 1000 == 0)
                                    count_logger.count(res_count);
                                if (checkpoints.count(res_count)) {
                                    TimePoint current_time = Clock::now();

                                    std::chrono::duration<double, std::milli> elapsed = current_time - start_time;

                                    double ms = elapsed.count();
                                    time_records.push_back({res_count, ms});
                                    cpu::recorder.record(to_string(res_count), ms);
                                    fmt::println("Output {}, cost Time: {} ms", res_count, ms);
                                }
                                if (OutputResults2)
                                    cout << temp_res << endl;
                                if (res_count >= total_num)
                                    return;
                            }
                            Btree.insert(temp_res);
                            new_FIX_X.clear();
                            new_new_X.clear();
                            new_new_X.insert(new_new_X.end(), X.begin(), X.end());
                            new_new_X.insert(new_new_X.end(), new_X.begin(), new_X.end());
                            new_FIX_X.insert(new_FIX_X.end(), FIX_X.begin(), FIX_X.end());
                            new_FIX_X.insert(new_FIX_X.end(), temp_FIX_X.begin(), temp_FIX_X.end());
                            Enumerate(new_new_X, F_Y, Full_Y, new_FIX_X, depth + 1, output + 1);
                            if (res_count >= total_num)
                                return;

                            if (output % 2) {
                                ++res_count;
                                if (res_count % 1000 == 0)
                                    count_logger.count(res_count);
                                if (checkpoints.count(res_count)) {
                                    TimePoint current_time = Clock::now();

                                    std::chrono::duration<double, std::milli> elapsed = current_time - start_time;

                                    double ms = elapsed.count();
                                    time_records.push_back({res_count, ms});
                                    cpu::recorder.record(to_string(res_count), ms);
                                    fmt::println("Output {}, cost Time: {} ms", res_count, ms);
                                }
                                if (OutputResults2)
                                    cout << temp_res << endl;
                                if (res_count >= total_num)
                                    return;
                            }
                        }
                    }

                    for (auto new_x: new_X) {
                        if (new_x not_eq pivot) {
                            G_index[new_x] = 0;
                            for (auto neighborOf_new_x: _Graph[new_x])
                                G_degree[neighborOf_new_x]--;
                        }
                    }
                }
                for (auto temp_fix_x: temp_FIX_X)
                    G_index[temp_fix_x] = 0;
            }

            s = min((int) (notPivotNeighbors_Renum.size()), Epsilon);
            notPivotPlexCount = notPivotNotPlexCount = 0;
            /*
             * 此时check_miss之前的代码没有动右边顶点的度，所以在子图里该是多少还是多少
             * notPivotPlexCount : 右边pivot没连接的顶点中，满足 k-1 biplex的个数
             * notPivotNotPlexCount : 不是k-1 biplex的个数
             * */
            for (auto notPivotNeighbor: notPivotNeighbors_Renum)
                if ((int) (X.size()) - G_degree[notPivotNeighbor] <= Epsilon - 1)
                    ++notPivotPlexCount;

            notPivotNotPlexCount = (int) notPivotNeighbors_Renum.size() - notPivotPlexCount;

            for (int emplaceIncCount = 1; emplaceIncCount <= s; ++emplaceIncCount) {
                /*
                 * 将非pivot邻居压入Inc中，依次压入
                 * */
                Inc.clear();
                for (temp_i = 0; temp_i < emplaceIncCount; ++temp_i)
                    Inc.emplace_back(notPivotNeighbors_Renum.begin() + temp_i);
                redo = true;
                while (redo) {
                    extension = true;
                    yu_exc = true;
                    for (auto inc: Inc) {
                        if (not(X.size() - G_degree[*inc] <= Epsilon - 1)) {
                            /*
                             * 如果inc的值(非pivot邻居)不满足k-1 biplex就不用extend了
                             * */
                            extension = false;
                            break;
                        }
                    }
                    if (extension) {
                        if ((emplaceIncCount < s and s < notPivotPlexCount) or
                            (s >= notPivotPlexCount and emplaceIncCount < notPivotPlexCount)) {
                            /*
                             * 不太懂这里
                             * */
                            redo = FindNext(notPivotNeighbors_Renum, Inc, emplaceIncCount);
                            continue;
                        }
                        temp_FIX_X.clear();
                        for (auto inc: Inc)
                            for (auto neighborOfInc: _Graph[*inc])
                                G_degree[neighborOfInc]++;
                        /*
                         * 之前把非pivot邻居的右边顶点剔除出子图，在这里重新加回来
                         * */

                        temp_v1 = (int) ((int) Y.size() - (int) notPivotNeighbors_Renum.size()) + emplaceIncCount;
                        for (auto neighborOfInc0: _Graph[*Inc[0]]) {
                            /*遍历inc0的邻居*/
                            if (G_index[neighborOfInc0] == 0 and G_degree[neighborOfInc0] == temp_v1 and
                                neighborOfInc0 not_eq pivot) {
                                /*
                                 * 此时左边的顶点中，原本H0的左边顶点G_index = 1
                                 * */
                                if (G_exc[neighborOfInc0] > 0) {
                                    extension = false;
                                    break;
                                }
                                temp_FIX_X.push_back(neighborOfInc0);
                                G_index[neighborOfInc0] = 2;
                            }
                        }

                        if (extension) {
                            for (auto inc: Inc)
                                G_index[*inc] = 1;
                            /*
                             * 在子图中把inc压入
                             * */
                            new_X.clear();
                            new_X.push_back(pivot);
                            X.push_back(pivot);
                            temp_check = checkY(X, emplaceIncCount, pivotNeighbors, Inc, FIX_X);
                            X.pop_back();
                            if (temp_check) {
                                temp_iter2 = pivotNeighbors.end();
                                temp_v1 = (int) ((int) Y.size() - (int) notPivotNeighbors_Renum.size()) - Epsilon +
                                          emplaceIncCount;
                                temp_v2 = (int) (X.size()) + (int) (new_X.size()) + 1 - Epsilon;
                                if (pivotNeighbors.size() + emplaceIncCount >= Epsilon + 1) {
                                    temp_check2 = true;
                                    for (auto inc: Inc) {
                                        for (auto neighborOfInc: _Graph[*inc]) {
                                            if (neighborOfInc != pivot && G_index[neighborOfInc] == 0 &&
                                                G_degree[neighborOfInc] >= temp_v1) {
                                                ext = true;
                                                for (auto twoHopNeighborOfInc: _Graph[neighborOfInc])
                                                    G_temp4[twoHopNeighborOfInc] = 1;
                                                for (auto neighbor: pivotNeighbors) {
                                                    if (!G_temp4[neighbor] && G_degree[neighbor] < temp_v2) {
                                                        ext = false;
                                                        break;
                                                    }
                                                }
                                                if (ext) {
                                                    for (auto inc: Inc) {
                                                        if (!G_temp4[*inc] and G_degree[*inc] < temp_v2) {
                                                            ext = false;
                                                            break;
                                                        }
                                                    }
                                                }
                                                for (auto twoHopNeighborOfInc: _Graph[neighborOfInc])
                                                    G_temp4[twoHopNeighborOfInc] = 0;
                                                if (ext) {
                                                    if (G_exc[neighborOfInc] > 0) {
                                                        yu_exc = false;
                                                        break;
                                                    }
                                                    new_X.push_back(neighborOfInc);
                                                    ++temp_v2;
                                                    G_index[neighborOfInc] = 1;
                                                    for (auto twoHopNeighborOfInc: _Graph[neighborOfInc])
                                                        G_degree[twoHopNeighborOfInc]++;
                                                }
                                            }
                                        }
                                        if (G_degree[*inc] == temp_v2 - 1) {
                                            temp_check2 = false;
                                            break;
                                        }
                                    }

                                    temp_count1 = temp_check ? emplaceIncCount : Epsilon + 1;
                                    for (auto neighbor: pivotNeighbors) {
                                        if (not(temp_count1 <= Epsilon && yu_exc))
                                            break;
                                        temp_count1++;
                                        for (auto twoHopNeighborOfPivot: _Graph[neighbor]) {
                                            if (twoHopNeighborOfPivot not_eq pivot and
                                                G_degree[twoHopNeighborOfPivot] == 0 and
                                                G_degree[twoHopNeighborOfPivot] >= temp_v1) {
                                                ext = true;
                                                for (auto threeHopNeighborOfPivot: _Graph[twoHopNeighborOfPivot])
                                                    G_temp4[threeHopNeighborOfPivot] = 1;
                                                for (auto _pivotNeighbor: pivotNeighbors) {
                                                    if (!G_temp4[_pivotNeighbor] &&
                                                        G_degree[_pivotNeighbor] < temp_v2) {
                                                        ext = false;
                                                        break;
                                                    }
                                                }
                                                if (ext) {
                                                    for (auto inc: Inc) {
                                                        if (!G_temp4[*inc] && G_degree[*inc] < temp_v2) {
                                                            ext = false;
                                                            break;
                                                        }
                                                    }
                                                }
                                                for (auto threeHopNeighborOfPivot: _Graph[twoHopNeighborOfPivot])
                                                    G_temp4[threeHopNeighborOfPivot] = 0;
                                                if (ext) {
                                                    if (G_exc[twoHopNeighborOfPivot] > 0) {
                                                        yu_exc = false;
                                                        break;
                                                    }
                                                    new_X.push_back(twoHopNeighborOfPivot);
                                                    ++temp_v2;
                                                    G_index[twoHopNeighborOfPivot] = 1;
                                                    for (auto threeHopNeighborOfPivot: _Graph[twoHopNeighborOfPivot])
                                                        G_degree[threeHopNeighborOfPivot]++;
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    temp_check2 = true;
                                    for (auto neighbor: pivotNeighbors) {
                                        if (G_degree[neighbor] == temp_v2 - 1) {
                                            temp_node1 = neighbor;
                                            temp_check2 = false;
                                            break;
                                        }
                                    }
                                    for (auto inc: Inc) {
                                        if (G_degree[*inc] == temp_v2 - 1) {
                                            temp_node1 = *inc;
                                            temp_check2 = false;
                                            break;
                                        }
                                    }
                                    if (!temp_check2) {
                                        for (auto neighborOf_temp_node1: _Graph[temp_node1]) {
                                            if (neighborOf_temp_node1 != pivot && G_index[neighborOf_temp_node1] == 0 &&
                                                G_degree[neighborOf_temp_node1] >= temp_v1) {
                                                ext = true;
                                                for (auto twoHopNeighbor: _Graph[neighborOf_temp_node1])
                                                    G_temp4[twoHopNeighbor] = 1;
                                                for (auto _pivotNeighbor: pivotNeighbors) {
                                                    if (!G_temp4[_pivotNeighbor] &&
                                                        G_degree[_pivotNeighbor] < temp_v2) {
                                                        ext = false;
                                                        break;
                                                    }
                                                }
                                                if (ext) {
                                                    for (auto inc: Inc) {
                                                        if (!G_temp4[*inc] and G_degree[*inc] < temp_v2) {
                                                            ext = false;
                                                            break;
                                                        }
                                                    }
                                                }
                                                for (auto twoHopNeighbor: _Graph[neighborOf_temp_node1])
                                                    G_temp4[twoHopNeighbor] = 0;
                                                if (ext) {
                                                    if (G_exc[neighborOf_temp_node1] > 0) {
                                                        yu_exc = false;
                                                        break;
                                                    }
                                                    new_X.push_back(neighborOf_temp_node1);
                                                    ++temp_v2;
                                                    G_index[neighborOf_temp_node1] = 1;
                                                    for (auto twoHopNeighbor: _Graph[neighborOf_temp_node1])
                                                        G_degree[twoHopNeighbor]++;
                                                }
                                            }
                                        }
                                    }
                                    for (int leftNode = 0; leftNode < Bipartite_index && temp_check2; ++leftNode) {
                                        if (leftNode != pivot && G_index[leftNode] == 0 &&
                                            G_degree[leftNode] >= temp_v1) {
                                            ext = true;
                                            for (auto neighborOfLeftNode: _Graph[leftNode])
                                                G_temp4[neighborOfLeftNode] = 1;
                                            for (temp_iter = pivotNeighbors.begin();
                                                 temp_iter != temp_iter2; ++temp_iter) {
                                                if (!G_temp4[*temp_iter] && G_degree[*temp_iter] < temp_v2) {
                                                    ext = false;
                                                    break;
                                                }
                                            }
                                            if (ext) {
                                                for (temp_i = 0; temp_i < emplaceIncCount; ++temp_i) {
                                                    temp_node2 = *Inc[temp_i];
                                                    if (!G_temp4[temp_node2] && G_degree[temp_node2] < temp_v2) {
                                                        ext = false;
                                                        break;
                                                    }
                                                }
                                            }
                                            for (auto neighborOfLeftNode: _Graph[leftNode])
                                                G_temp4[neighborOfLeftNode] = 0;
                                            if (ext) {
                                                if (G_exc[leftNode] > 0) {
                                                    yu_exc = false;
                                                    break;
                                                }
                                                new_X.push_back(leftNode);
                                                ++temp_v2;
                                                G_index[leftNode] = 1;
                                                for (temp_j = degree[leftNode] - 1; temp_j >= 0; --temp_j) {
                                                    G_degree[_Graph[leftNode][temp_j]]++;
                                                }
                                            }
                                        }
                                    }
                                }

                                if (yu_exc) {
                                    F_Y.clear();
                                    X_a.clear();
                                    X_a.insert(X_a.end(), X.begin(), X.end());
                                    X_a.insert(X_a.end(), new_X.begin(), new_X.end());
                                    X_a.insert(X_a.end(), FIX_X.begin(), FIX_X.end());
                                    X_a.insert(X_a.end(), temp_FIX_X.begin(), temp_FIX_X.end());
                                    F_Y.insert(F_Y.end(), pivotNeighbors.begin(), pivotNeighbors.end());
                                    for (auto inc: Inc)
                                        F_Y.emplace_back(*inc);

                                    string temp_res;
                                    TransToString2(X_a, F_Y, temp_res);
                                    if (Btree.find(temp_res) == Btree.end()) {
                                        if (output % 2 == 0) {
                                            ++res_count;
                                            if (res_count % 1000 == 0)
                                                count_logger.count(res_count);
                                            if (checkpoints.count(res_count)) {
                                                TimePoint current_time = Clock::now();

                                                std::chrono::duration<double, std::milli> elapsed =
                                                        current_time - start_time;

                                                double ms = elapsed.count();
                                                time_records.push_back({res_count, ms});
                                                cpu::recorder.record(to_string(res_count), ms);
                                                fmt::println("Output {}, cost Time: {} ms", res_count, ms);
                                            }
                                            if (OutputResults2)
                                                cout << temp_res << endl;
                                            if (res_count >= total_num)
                                                return;
                                        }
                                        Btree.insert(temp_res);
                                        new_FIX_X.clear();
                                        new_new_X.clear();
                                        new_new_X.insert(new_new_X.end(), X.begin(), X.end());
                                        new_new_X.insert(new_new_X.end(), new_X.begin(), new_X.end());
                                        new_FIX_X.insert(new_FIX_X.end(), FIX_X.begin(), FIX_X.end());
                                        new_FIX_X.insert(new_FIX_X.end(), temp_FIX_X.begin(), temp_FIX_X.end());

                                        Enumerate(new_new_X, F_Y, Full_Y, new_FIX_X, depth + 1, output + 1);
                                        if (res_count >= total_num)
                                            return;

                                        if (output % 2) {
                                            ++res_count;
                                            if (res_count % 1000 == 0)
                                                count_logger.count(res_count);
                                            if (checkpoints.count(res_count)) {
                                                TimePoint current_time = Clock::now();

                                                std::chrono::duration<double, std::milli> elapsed =
                                                        current_time - start_time;

                                                double ms = elapsed.count();
                                                time_records.push_back({res_count, ms});
                                                cpu::recorder.record(to_string(res_count), ms);
                                                fmt::println("Output {}, cost Time: {} ms", res_count, ms);
                                            }
                                            if (OutputResults2)
                                                cout << temp_res << endl;
                                            if (res_count >= total_num)
                                                return;
                                        }
                                    }
                                }
                            }

                            for (auto x: new_X) {
                                if (x != pivot) {
                                    G_index[x] = 0;
                                    for (auto neighborOfX: _Graph[x])
                                        G_degree[neighborOfX]--;
                                }
                            }
                            for (auto inc: Inc)
                                G_index[*inc] = 0;
                        }
                        for (auto inc: Inc)
                            for (auto neighborOfInc: _Graph[*inc])
                                G_degree[neighborOfInc]--;
                        for (auto temp_fix_x: temp_FIX_X)
                            G_index[temp_fix_x] = 0;
                        for (auto inc: Inc)
                            G_index[*inc] = 0;
                    } else {
                        if ((notPivotPlexCount >= s && emplaceIncCount < s) or (notPivotPlexCount < s) and
                            (emplaceIncCount <= notPivotPlexCount)) {
                            redo = FindNext(notPivotNeighbors_Renum, Inc, emplaceIncCount);
                            continue;
                        }
                        extension = true;
                        temp_FIX_X.clear();
                        for (auto inc: Inc)
                            for (auto neighborOfInc: _Graph[*inc])
                                G_degree[neighborOfInc]++;

                        temp_node1 = *Inc[0];
                        temp_v1 = (int) (Y.size() - notPivotNeighbors_Renum.size()) + emplaceIncCount;
                        for (auto neighborOfInc0: _Graph[*Inc[0]]) {
                            if (G_index[neighborOfInc0] == 0 and G_degree[neighborOfInc0] == temp_v1) {
                                if (G_exc[neighborOfInc0]) {
                                    extension = false;
                                    break;
                                }
                                temp_FIX_X.push_back(neighborOfInc0);
                                G_index[neighborOfInc0] = 2;
                            }
                        }

                        if (extension) {
                            NB.clear();
                            for (auto inc: Inc)
                                for (auto neighborOfInc: _Graph[*inc])
                                    G_temp2[neighborOfInc]++;

                            for (auto x: X)
                                if (G_temp2[x] not_eq emplaceIncCount)
                                    NB.emplace_back(x);

                            for (auto inc: Inc)
                                for (auto neighborOfInc: _Graph[*inc])
                                    G_temp2[neighborOfInc]++;

                            NB_S = min((int) NB.size(), s);

                            for (int k = 1; k <= NB_S; ++k) {
                                FIX.clear();
                                for (temp_i = 0; temp_i < k; ++temp_i)
                                    FIX.emplace_back(NB.begin() + temp_i);

                                redo2 = true;
                                while (redo2) {
                                    for (auto fix: FIX) {
                                        for (auto neighborOfFix: _Graph[*fix])
                                            G_degree[neighborOfFix]--;
                                        G_index[*fix] = 0;
                                        X.erase(find(X.begin(), X.end(), *fix));
                                    }
                                    extension = true;
                                    temp_v1 = (int) X.size() + 1 - Epsilon;
                                    for (auto inc: Inc) {
                                        if (G_degree[*inc] < temp_v1) {
                                            extension = false;
                                            break;
                                        }
                                    }

                                    if (extension) {
                                        for (auto inc: Inc)
                                            G_index[*inc] = 1;
                                        new_X.clear();
                                        new_X.push_back(pivot);
                                        X.push_back(pivot);
                                        temp_check = checkY(X, emplaceIncCount, pivotNeighbors, Inc, FIX_X);
                                        X.pop_back();
                                        if (temp_check) {
                                            yu_exc = true;
                                            temp_iter2 = pivotNeighbors.end();
                                            temp_v1 = (int) (Y.size() - notPivotNeighbors_Renum.size()) - Epsilon +
                                                      emplaceIncCount;
                                            temp_v2 = (int) (X.size()) + (int) (new_X.size()) + 1 - Epsilon;
                                            if (pivotNeighbors.size() + emplaceIncCount >= Epsilon + 1) {
                                                temp_check2 = true;
                                                for (auto inc: Inc) {
                                                    for (auto neighborOfInc: _Graph[*inc]) {
                                                        if (neighborOfInc != pivot && G_index[neighborOfInc] == 0 &&
                                                            G_degree[neighborOfInc] >= temp_v1) {
                                                            ext = true;
                                                            for (auto twoHopNeighborOfInc: _Graph[neighborOfInc])
                                                                G_temp4[twoHopNeighborOfInc] = 1;
                                                            for (auto _pivotNeighbor: pivotNeighbors) {
                                                                if (!G_temp4[_pivotNeighbor] &&
                                                                    G_degree[_pivotNeighbor] < temp_v2) {
                                                                    ext = false;
                                                                    break;
                                                                }
                                                            }
                                                            if (ext) {
                                                                for (auto _inc: Inc) {
                                                                    if (!G_temp4[*inc] && G_degree[*inc] < temp_v2) {
                                                                        ext = false;
                                                                        break;
                                                                    }
                                                                }
                                                            }
                                                            for (auto twoHopNeighborOfInc: _Graph[neighborOfInc])
                                                                G_temp4[twoHopNeighborOfInc] = 0;
                                                            if (ext) {
                                                                if (G_exc[neighborOfInc] > 0) {
                                                                    yu_exc = false;
                                                                    break;
                                                                }
                                                                new_X.push_back(neighborOfInc);
                                                                ++temp_v2;
                                                                G_index[neighborOfInc] = 1;
                                                                for (auto twoHopNeighborOfInc: _Graph[neighborOfInc])
                                                                    G_degree[twoHopNeighborOfInc]++;
                                                            }
                                                        }
                                                    }
                                                    if (G_degree[*inc] == temp_v2 - 1) {
                                                        temp_check2 = false;
                                                        break;
                                                    }
                                                }

                                                temp_count1 = temp_check2 ? emplaceIncCount : Epsilon + 1;
                                                for (auto _pivotNeighbor: pivotNeighbors) {
                                                    if (not(temp_count1 < Epsilon + 1 && yu_exc))break;
                                                    ++temp_count1;
                                                    for (auto twoHopNeighborOfPivot: _Graph[_pivotNeighbor]) {
                                                        if (twoHopNeighborOfPivot != pivot &&
                                                            G_index[twoHopNeighborOfPivot] == 0 &&
                                                            G_degree[twoHopNeighborOfPivot] >= temp_v1) {
                                                            ext = true;
                                                            for (auto threeHopNeighborOfPivot: _Graph[twoHopNeighborOfPivot])
                                                                G_temp4[threeHopNeighborOfPivot] = 1;

                                                            for (auto _pivotNeighbor_: pivotNeighbors) {
                                                                if (!G_temp4[_pivotNeighbor_] &&
                                                                    G_degree[_pivotNeighbor_] < temp_v2) {
                                                                    ext = false;
                                                                    break;
                                                                }
                                                            }
                                                            if (ext) {
                                                                for (auto inc: Inc) {
                                                                    if (!G_temp4[*inc] && G_degree[*inc] < temp_v2) {
                                                                        ext = false;
                                                                        break;
                                                                    }
                                                                }
                                                            }
                                                            for (auto threeHopNeighborOfPivot: _Graph[twoHopNeighborOfPivot])
                                                                G_temp4[threeHopNeighborOfPivot] = 1;
                                                            if (ext) {
                                                                if (G_exc[twoHopNeighborOfPivot] > 0) {
                                                                    yu_exc = false;
                                                                    break;
                                                                }
                                                                ++temp_v2;
                                                                new_X.push_back(twoHopNeighborOfPivot);
                                                                G_index[twoHopNeighborOfPivot] = 1;
                                                                for (auto threeHopNeighborOfPivot: _Graph[twoHopNeighborOfPivot])
                                                                    G_degree[threeHopNeighborOfPivot]++;
                                                            }
                                                        }
                                                    }
                                                }
                                                for (auto _pivotNeighbor: pivotNeighbors) {
                                                    if (not(temp_count1 < Epsilon + 1 && yu_exc))
                                                        break;
                                                    temp_count1++;
                                                    for (auto twoHopNeighborOfPivot: _Graph[_pivotNeighbor]) {
                                                        if (twoHopNeighborOfPivot != pivot &&
                                                            G_index[twoHopNeighborOfPivot] == 0 &&
                                                            G_degree[twoHopNeighborOfPivot] >= temp_v1) {
                                                            ext = true;
                                                            for (auto threeHopNeighborOfPivot: _Graph[twoHopNeighborOfPivot])
                                                                G_temp4[threeHopNeighborOfPivot] = 1;
                                                            for (auto _pivotNeighbor_: pivotNeighbors) {
                                                                if (!G_temp4[_pivotNeighbor_] &&
                                                                    G_degree[_pivotNeighbor_] < temp_v2) {
                                                                    ext = false;
                                                                    break;
                                                                }
                                                            }
                                                            if (ext) {
                                                                for (auto inc: Inc) {
                                                                    if (!G_temp4[*inc] && G_degree[*inc] < temp_v2) {
                                                                        ext = false;
                                                                        break;
                                                                    }
                                                                }
                                                            }
                                                            for (auto threeHopNeighborOfPivot: _Graph[twoHopNeighborOfPivot])
                                                                G_temp4[threeHopNeighborOfPivot] = 0;

                                                            if (ext) {
                                                                if (G_exc[twoHopNeighborOfPivot] > 0) {
                                                                    yu_exc = false;
                                                                    break;
                                                                }
                                                                ++temp_v2;
                                                                new_X.push_back(twoHopNeighborOfPivot);
                                                                G_index[twoHopNeighborOfPivot] = 1;
                                                                for (auto threeHopNeighborOfPivot: _Graph[twoHopNeighborOfPivot])
                                                                    G_degree[threeHopNeighborOfPivot]++;
                                                            }
                                                        }
                                                    }
                                                }
                                            } else {
                                                temp_check2 = true;
                                                for (auto _pivotNeighbor: pivotNeighbors) {
                                                    if (G_degree[_pivotNeighbor] == temp_v2 - 1) {
                                                        temp_check2 = false;
                                                        temp_node1 = _pivotNeighbor;
                                                        break;
                                                    }
                                                }
                                                for (auto inc: Inc) {
                                                    if (G_degree[*inc] == temp_v2 - 1) {
                                                        temp_check2 = false;
                                                        temp_node1 = *inc;
                                                        break;
                                                    }
                                                }
                                                if (!temp_check2) {
                                                    for (auto neighborOf_temp_node1: _Graph[temp_node1]) {
                                                        if (neighborOf_temp_node1 != pivot &&
                                                            G_index[neighborOf_temp_node1] == 0 &&
                                                            G_degree[neighborOf_temp_node1] >= temp_v1) {
                                                            ext = true;
                                                            for (auto twoHopNeighbor: _Graph[neighborOf_temp_node1])
                                                                G_temp4[twoHopNeighbor] = 1;
                                                            for (auto _pivotNeighbor_: pivotNeighbors) {
                                                                if (!G_temp4[_pivotNeighbor_] &&
                                                                    G_degree[_pivotNeighbor_] < temp_v2) {
                                                                    ext = false;
                                                                    break;
                                                                }
                                                            }
                                                            if (ext) {
                                                                for (auto inc: Inc) {
                                                                    if (!G_temp4[*inc] && G_degree[*inc] < temp_v2) {
                                                                        ext = false;
                                                                        break;
                                                                    }
                                                                }
                                                            }
                                                            for (auto twoHopNeighbor: _Graph[neighborOf_temp_node1])
                                                                G_temp4[twoHopNeighbor] = 0;
                                                            if (ext) {
                                                                if (G_exc[neighborOf_temp_node1] > 0) {
                                                                    yu_exc = false;
                                                                    break;
                                                                }
                                                                ++temp_v2;
                                                                new_X.push_back(neighborOf_temp_node1);
                                                                G_index[neighborOf_temp_node1] = 1;
                                                                for (auto twoHopNeighbor: _Graph[neighborOf_temp_node1])
                                                                    G_degree[twoHopNeighbor]++;
                                                            }
                                                        }
                                                    }
                                                }
                                                for (int leftNodes = 0;
                                                     leftNodes < Bipartite_index and temp_check2; leftNodes++) {
                                                    if (leftNodes not_eq pivot and G_index[leftNodes] == 0 and
                                                        G_degree[leftNodes] >= temp_v1) {
                                                        ext = true;
                                                        for (auto neighborOfLeftNode: _Graph[leftNodes])
                                                            G_temp4[neighborOfLeftNode] = 1;

                                                        for (auto pivotNeighbor: pivotNeighbors) {
                                                            if (not G_temp4[pivotNeighbor] and
                                                                G_degree[pivotNeighbor] < temp_v2) {
                                                                ext = false;
                                                                break;
                                                            }
                                                        }

                                                        if (ext) {
                                                            for (auto inc: Inc) {
                                                                if (not G_temp4[*inc] and G_degree[*inc] < temp_v2) {
                                                                    ext = false;
                                                                    break;
                                                                }
                                                            }
                                                        }

                                                        for (auto neighborOfLeftNode: _Graph[leftNodes])
                                                            G_temp4[neighborOfLeftNode] = 0;

                                                        if (ext) {
                                                            if (G_exc[leftNodes]) {
                                                                yu_exc = false;
                                                                break;
                                                            }
                                                            ++temp_v2;
                                                            new_X.push_back(leftNodes);
                                                            G_index[leftNodes] = 1;
                                                            for (auto neighborOfLeftNode: _Graph[leftNodes])
                                                                G_degree[neighborOfLeftNode]++;
                                                        }
                                                    }
                                                }
                                            }
                                            if (yu_exc) {
                                                F_Y.clear();
                                                X_a.clear();
                                                X_a.insert(X_a.end(), X.begin(), X.end());
                                                X_a.insert(X_a.end(), new_X.begin(), new_X.end());
                                                X_a.insert(X_a.end(), FIX_X.begin(), FIX_X.end());
                                                X_a.insert(X_a.end(), temp_FIX_X.begin(), temp_FIX_X.end());
                                                F_Y.insert(F_Y.end(), pivotNeighbors.begin(), pivotNeighbors.end());
                                                for (auto inc: Inc)
                                                    F_Y.emplace_back(*inc);

                                                string temp_res;
                                                TransToString2(X_a, F_Y, temp_res);
                                                if (Btree.find(temp_res) == Btree.end()) {
                                                    if (output % 2 == 0) {
                                                        ++res_count;
                                                        if (res_count % 1000 == 0)
                                                            count_logger.count(res_count);
                                                        if (checkpoints.count(res_count)) {
                                                            TimePoint current_time = Clock::now();

                                                            std::chrono::duration<double, std::milli> elapsed =
                                                                    current_time - start_time;

                                                            double ms = elapsed.count();
                                                            time_records.push_back({res_count, ms});
                                                            cpu::recorder.record(to_string(res_count), ms);
                                                            fmt::println("Output {}, cost Time: {} ms", res_count, ms);
                                                        }
                                                        if (OutputResults2)
                                                            cout << temp_res << endl;
                                                        if (res_count >= total_num)
                                                            return;
                                                    }
                                                    Btree.insert(temp_res);
                                                    new_FIX_X.clear();
                                                    new_new_X.clear();
                                                    new_new_X.insert(new_new_X.end(), X.begin(), X.end());
                                                    new_new_X.insert(new_new_X.end(), new_X.begin(), new_X.end());
                                                    new_FIX_X.insert(new_FIX_X.end(), FIX_X.begin(), FIX_X.end());
                                                    new_FIX_X.insert(new_FIX_X.end(), temp_FIX_X.begin(),
                                                                     temp_FIX_X.end());

                                                    Enumerate(new_new_X, F_Y, Full_Y, new_FIX_X, depth + 1, output + 1);
                                                    if (res_count >= total_num)
                                                        return;
                                                    if (output % 2) {
                                                        ++res_count;
                                                        if (res_count % 1000 == 0)
                                                            count_logger.count(res_count);
                                                        if (checkpoints.count(res_count)) {
                                                            TimePoint current_time = Clock::now();

                                                            std::chrono::duration<double, std::milli> elapsed =
                                                                    current_time - start_time;

                                                            double ms = elapsed.count();
                                                            time_records.push_back({res_count, ms});
                                                            cpu::recorder.record(to_string(res_count), ms);
                                                            fmt::println("Output {}, cost Time: {} ms", res_count, ms);
                                                        }
                                                        if (OutputResults2)
                                                            cout << temp_res << endl;
                                                        if (res_count >= total_num)
                                                            return;
                                                    }
                                                }
                                            }
                                        }
                                        for (auto new_x: new_X) {
                                            if (new_x not_eq pivot) {
                                                G_index[new_x] = 0;
                                                for (auto neighborOf_new_x: _Graph[new_x])
                                                    G_degree[neighborOf_new_x]--;
                                            }
                                        }
                                        for (auto inc: Inc)
                                            G_index[*inc] = 0;
                                    }
                                    for (auto fix: FIX) {
                                        for (auto neighborOf_fix: _Graph[*fix])
                                            G_degree[neighborOf_fix]++;
                                        G_index[*fix] = 1;
                                        X.emplace_back(*fix);
                                    }
                                    for (auto temp_fix_x: temp_FIX_X)
                                        G_index[temp_fix_x] = 0;
                                    temp_FIX_X.clear();
                                    redo2 = FindNext(NB, FIX, k);
                                }
                            }
                        }
                        for (auto inc: Inc)
                            for (auto neighborOfInc: _Graph[*inc])
                                G_degree[neighborOfInc]--;
                        for (auto temp_fix_x: temp_FIX_X)
                            G_index[temp_fix_x] = 0;
                        for (auto inc: Inc)
                            G_index[*inc] = 0;
                    }
                    redo = FindNext(notPivotNeighbors_Renum, Inc, emplaceIncCount);
                }
            }

            G_index[pivot] = 0;
            for (auto neighborOfPivot: _Graph[pivot])
                G_degree[neighborOfPivot]--;
            for (auto _notPivotNeighbors: notPivotNeighbors_Renum) {
                for (auto neighborOf_notPivotNeighbors: _Graph[_notPivotNeighbors])
                    G_degree[neighborOf_notPivotNeighbors]++;
                G_index[_notPivotNeighbors] = 1;
            }
            if (G_exc[pivot] == 0)
                G_exc[pivot] = depth;
        }
        for (int leftNode = 0; leftNode < Bipartite_index; leftNode++)
            if (G_exc[leftNode] == depth)
                G_exc[leftNode] = 0;
    }

    bool
    iTraversal::checkY(vector<int> &X, int yuj, vector<int> &cons_Y, vector<int>::iterator Inc[], vector<int> &FIX_X) {
        temp_check1 = true;
        temp_check2 = true;
        temp_v1 = (int) (X.size());
        temp_v2 = (int) (cons_Y.size());
        temp_iter3 = X.end();
        temp_iter2 = FIX_X.end();

        for (temp_i = 0; temp_i < yuj; ++temp_i) {
            G_temp3[*Inc[temp_i]] = 1;
        }

        if (temp_v1 >= Epsilon + 1) {
            temp_count1 = 0;
            for (temp_iter = X.begin(); temp_iter != temp_iter3 && temp_count1 < Epsilon + 1; ++temp_iter) {
                ++temp_count1;
                for (temp_i = 0; temp_i < degree[*temp_iter]; ++temp_i) {

                    temp_node1 = _Graph[*temp_iter][temp_i];
                    if (G_index[temp_node1] == 0 && G_degree[temp_node1] >= temp_v1 - Epsilon && !G_temp3[temp_node1]) {
                        temp_check1 = false;
                        for (temp_yu = degree[temp_node1] - 1; temp_yu >= 0; --temp_yu) {
                            G_temp4[_Graph[temp_node1][temp_yu]] = 1;
                        }
                        for (temp_iter1 = X.begin(); temp_iter1 != temp_iter3; ++temp_iter1) {
                            if (!G_temp4[*temp_iter1] && G_degree[*temp_iter1] < temp_v2 + yuj - Epsilon + 1) {
                                temp_check1 = true;
                                break;
                            }
                        }
                        if (!temp_check1 && not FIX_X.empty()) {
                            temp_count2 = 0;
                            for (temp_iter1 = FIX_X.begin(); temp_iter1 != temp_iter2; ++temp_iter1) {
                                if (!G_temp4[*temp_iter1]) {
                                    temp_count2++;
                                }
                            }
                            if (G_degree[temp_node1] < temp_v1 - Epsilon + temp_count2) {
                                temp_check1 = true;
                            }
                        }
                        for (temp_yu = degree[temp_node1] - 1; temp_yu >= 0; --temp_yu) {
                            G_temp4[_Graph[temp_node1][temp_yu]] = 0;
                        }
                        if (!temp_check1) {
                            break;
                        }
                    }
                }
                if (!temp_check1)
                    break;
            }
        } else {
            for (temp_node1 = Bipartite_index; temp_node1 < Graph_size; ++temp_node1) {
                if (G_index[temp_node1] == 0 && G_degree[temp_node1] >= temp_v1 - Epsilon && !G_temp3[temp_node1]) {
                    temp_check1 = false;
                    for (temp_yu = degree[temp_node1] - 1; temp_yu >= 0; --temp_yu) {
                        G_temp4[_Graph[temp_node1][temp_yu]] = 1;
                    }
                    for (temp_iter1 = X.begin(); temp_iter1 != temp_iter3; ++temp_iter1) {
                        if (!G_temp4[*temp_iter1] && G_degree[*temp_iter1] < temp_v2 + yuj - Epsilon + 1) {
                            temp_check1 = true;
                            break;
                        }
                    }
                    if (!temp_check1 && not FIX_X.empty()) {
                        temp_count2 = 0;
                        for (temp_iter1 = FIX_X.begin(); temp_iter1 != temp_iter2; ++temp_iter1) {
                            if (!G_temp4[*temp_iter1]) {
                                ++temp_count2;
                            }
                        }
                        if (G_degree[temp_node1] < temp_v1 - Epsilon + temp_count2) {
                            temp_check1 = true;
                        }
                    }
                    for (temp_yu = degree[temp_node1] - 1; temp_yu >= 0; --temp_yu) {
                        G_temp4[_Graph[temp_node1][temp_yu]] = 0;
                    }
                    if (!temp_check1) {
                        break;
                    }
                }
            }
        }

        for (temp_i = 0; temp_i < yuj; ++temp_i) {
            G_temp3[*Inc[temp_i]] = 0;
        }
        return temp_check1;
    }

    bool
    iTraversal::checkY(vector<int> &X, int emplaceIncCount, vector<int> &pivotNeighbor,
                       vector <vector<int>::iterator> Inc, vector<int> &FIX_X) {
        temp_check1 = true;
        temp_check2 = true;
        for (auto inc: Inc)
            G_temp3[*inc] = 1;
        /*
         * inc的值（右边）已经被压入子图，不需要在压入
         * */

        if ((int) X.size() >= Epsilon + 1) {
            /*
             * 如果左边顶点数量大于等于k + 1，则可以减少右边顶点的查找范围
             * 不知道为什么
             * */
            temp_count1 = 0;
            for (auto x: X) {
                if (temp_count1 >= Epsilon + 1)
                    break;
                ++temp_count1;
                for (auto neighborOfX: _Graph[x]) {
                    if (G_index[neighborOfX] == 0 and (int) X.size() - G_degree[neighborOfX] <= Epsilon and
                        !G_temp3[neighborOfX]) {
                        /*
                         * 把检查Y顶点的范围限制在k+1个X顶点的邻居范围内，这些右边顶点满足：
                         * 条件1：不在H0中
                         * 条件2：在X组成的子图中，满足biplex，|X| - |N(neighborOfX)| <= k 是biplex的定义
                         * 条件3：不在inc中
                         * */
                        temp_check1 = false;
                        for (auto towHopNeighborOfX: _Graph[neighborOfX])
                            G_temp4[towHopNeighborOfX] = 1;
                        /*标记这些右边顶点的邻居*/

                        for (auto _x: X) {
                            if (!G_temp4[_x] &&
                                G_degree[_x] < (int) pivotNeighbor.size() + emplaceIncCount - Epsilon + 1) {
                                /*
                                 *
                                 * 条件2：对(int)pivotNeighbor.size() + emplaceIncCount - G_degree[_x] <= Epsilon - 1 取反
                                 * 理解：在非右边顶点邻居中，如果有任意一个顶点在pivot neighbor和inc组成的子图中不满足k-1 biplex，则temp_ckeck1 = true
                                 * */
                                temp_check1 = true;
                                break;
                            }
                        }
                        if (temp_check1 == false && not FIX_X.empty()) {
                            /*
                             * 如果前面X不满足条件，则检查FIX_X是否满足条件
                             * */
                            /* 统计 FIX_X中不是右边顶点的邻居*/
                            temp_count2 = 0;
                            for (auto fix_x: FIX_X)
                                if (!G_temp4[fix_x]) temp_count2++;

                            /*
                             * |X| + 非右边顶点邻居数量 - 右边顶点邻居数 > Epsilon
                             * 如果有任意一个右边顶点，在X和非右边顶点邻居组成的子图中，不满足k-biplex，则temp_check1 = true
                             * */
                            if (G_degree[neighborOfX] < (int) X.size() - Epsilon + temp_count2)
                                temp_check1 = true;
                        }
                        for (auto towHopNeighborOfX: _Graph[neighborOfX])
                            G_temp4[towHopNeighborOfX] = 0;
                        if (!temp_check1)
                            break;
                    }
                }
            }
        } else {
            /*
             * 这个条件分支的其实是|X| <= k, 因为|X| < k+1在整数上等价于|X| <= k
             * 那么|X| - 任意 <= k, 必然满足k-biplex
             * */
            for (int rightNodes = Bipartite_index; rightNodes < Bipartite_index; rightNodes++) {
                if (G_index[rightNodes] == 0 and (int) X.size() - G_degree[rightNodes] <= Epsilon and
                    !G_temp3[rightNodes]) {
                    /*
                     * 对于所有的右边顶点，是否满足：
                     * 1. 不在H0内
                     * 2. 在现在的X中，满足k-biplex的定义
                     * 3. 不在inc中
                     * */
                    temp_check1 = false;
                    for (auto neighborOfRight: _Graph[rightNodes])
                        G_temp4[neighborOfRight] = 1;
                    for (auto x: X) {
                        if (!G_temp4[x] && G_degree[x] < (int) pivotNeighbor.size() + emplaceIncCount - Epsilon + 1) {
                            temp_check1 = true;
                            break;
                        }
                    }
                    if (!temp_check1 && not FIX_X.empty()) {
                        temp_count2 = 0;
                        for (auto fix_x: FIX_X)
                            if (!G_temp4[fix_x])
                                ++temp_count2;
                        if (G_degree[rightNodes] < (int) X.size() - Epsilon + temp_count2)
                            temp_check1 = true;
                    }
                    for (auto neighborOfRight: _Graph[rightNodes])
                        G_temp4[neighborOfRight] = 0;
                    if (!temp_check1)
                        break;
                }
            }
        }

        for (auto inc: Inc)
            G_temp3[*inc] = 0;
        return temp_check1;
    }
};
#endif