#include "biplex.h"
#include "bigraph.hpp"
#include "cmdline.hpp"
#include "load_bipartite.hpp"
#include "search_logger.hpp"
#include <iostream>

#define RESULT_MAXNUM 0xFFFFFFFFFFFFFFFFLLU

int main(int argc, char* argv[]) {

	cmdline::parser args;

	args.add<std::string>("data", 'd', "dataset path", true);
	args.add<int>("lb", 'q', "size lowerbound", true);
	args.add<int>("key", 'k', "value of k", true);
	args.add<unsigned long long>("num", 'n', "number of result", false, RESULT_MAXNUM);
	args.add<int>("output", 'o', "output biplex vertex sets (1=yes, 0=no)", false, 0);
	args.add("no-upperbound", 0, "disable upperbound pruning");
	args.add("no-pivoting", 0, "disable pivoting (i.e. using algorithm BPBnB instead of BPPivot)");
	args.add("no-core-reduction", 0, "disable core reduction");
	args.add("no-butterfly-reduction", 0, "disable butterfly reduction");
	args.add("no-ordering", 0, "disable ordering");

	args.parse_check(argc, argv);

    BiGraph G = loadBipartiteGraphFromBin(args.get<std::string>("data"));
	SearchLogger logger(args.get<std::string>("data"), args.get<int>("key"));
	biplex::Result result = biplex::run(
		G,
		args.get<int>("lb"),
		args.get<int>("key"),
		args.get<unsigned long long>("num"),
		!args.exist("no-upperbound"),
		!args.exist("no-pivoting"),
		!args.exist("no-core-reduction"),
		!args.exist("no-butterfly-reduction"),
		!args.exist("no-ordering"),
		args.get<int>("output") != 0,
		&logger
	);

	biplex::printResult(result);

	return 0;
}
