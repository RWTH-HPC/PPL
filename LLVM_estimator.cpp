#include <llvm/IR/Module.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>

#include <sys/resource.h>
#include <stdio.h>

// Compile with: clang++ LLVM_estimator.cpp -lpthread `llvm-config --libs` `llvm-config --ldflags` -ltinfo

int main(int argc, char *argv[]) {
  llvm::LLVMContext context;
  llvm::SMDiagnostic error;

  struct rusage before, first, second, after;
  getrusage(RUSAGE_SELF, &before);

  auto m1 = llvm::parseIRFile(argv[1], error, context);
  getrusage(RUSAGE_SELF, &first);

  {
    auto m2 = llvm::parseIRFile(argv[1], error, context);
    getrusage(RUSAGE_SELF, &second);
  }
  getrusage(RUSAGE_SELF, &after);

  // printf("max: %ld %ld %ld %ld\n", before.ru_maxrss, first.ru_maxrss, second.ru_maxrss, after.ru_maxrss);
  long int size = second.ru_maxrss - first.ru_maxrss;
  printf("%ld bytes\n", size * 1024);

  return 0;
}
