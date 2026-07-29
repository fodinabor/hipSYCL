// LLVM function pass to remove all barrier calls.
//
// Copyright (c) 2016 Pekka Jääskeläinen / TUT
//               2021 Aksel Alpay and hipSYCL contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "hipSYCL/compiler/cbs/RemoveBarrierCalls.hpp"

#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/compiler/cbs/SplitterAnnotationAnalysis.hpp"

#include <llvm/IR/Constants.h>
#include <llvm/Support/Casting.h>

namespace hipsycl::compiler {
namespace {

using namespace cbs;

// Removes the loads of a CBS pseudo global variable and, once unused, the variable
// itself. Loads with remaining uses can only be replaced with poison in non-kernel
// functions: those are leftover, never-called copies of libkernel functions that were
// force-inlined into the kernels. Kernels that have not been processed by
// SubCfgFormation yet must keep their loads (their uses are replaced there).
bool deleteGlobalVariable(llvm::Module *M, llvm::StringRef VarName,
                          const SplitterAnnotationInfo &SAA) {
  auto *GV = M->getGlobalVariable(VarName);
  if (!GV)
    return false;

  llvm::SmallVector<llvm::LoadInst *, 8> Loads;
  for (auto U : GV->users())
    if (auto LI = llvm::dyn_cast<llvm::LoadInst>(U))
      Loads.push_back(LI);

  bool Changed = false;
  for (auto *LI : Loads) {
    if (!LI->user_empty()) {
      if (SAA.isKernelFunc(LI->getFunction()))
        continue;
      HIPSYCL_DEBUG_INFO << "[RemoveBarrierCalls] Replace leftover load " << *LI << " in "
                         << LI->getFunction()->getName() << " with poison\n";
      LI->replaceAllUsesWith(llvm::PoisonValue::get(LI->getType()));
    }
    LI->eraseFromParent();
    Changed = true;
  }

  if (GV->use_empty()) {
    HIPSYCL_DEBUG_INFO << "[RemoveBarrierCalls] Clean-up global variable " << *GV << "\n";
    GV->eraseFromParent();
    return true;
  }
  HIPSYCL_DEBUG_INFO << "[RemoveBarrierCalls] Global variable still in use " << VarName << "\n";
  for (auto *U : GV->users()) {
    HIPSYCL_DEBUG_INFO << "[RemoveBarrierCalls] >>> " << *U;
    if (auto I = llvm::dyn_cast<llvm::Instruction>(U)) {
      HIPSYCL_DEBUG_EXECUTE_INFO(
        llvm::outs() << " in " << I->getFunction()->getName()
      );
    }
    HIPSYCL_DEBUG_EXECUTE_INFO(llvm::outs() << "\n");
  }
  return Changed;
}

// Leftover calls to the __cbs_* pseudo intrinsics (reduce/shuffle/broadcast/...) in
// non-kernel functions are replaced with their data operand; kernels have them lowered
// by SubCfgFormation. This is only correct for sub-groups of size 1, but the only
// surviving calls are in dead copies of functions that were inlined into the kernels.
// Without this cleanup they would produce undefined-symbol errors at host link time.
bool removeLeftoverCBSIntrinsics(llvm::Module *M, const SplitterAnnotationInfo &SAA) {
  llvm::SmallVector<llvm::Function *, 8> Intrinsics;
  for (auto &F : *M)
    if (F.isDeclaration() && utils::isCBSIntrinsic(&F))
      Intrinsics.push_back(&F);

  bool Changed = false;
  for (auto *F : Intrinsics) {
    llvm::SmallVector<llvm::CallInst *, 8> Calls;
    for (auto *U : F->users())
      if (auto *CI = llvm::dyn_cast<llvm::CallInst>(U);
          CI && CI->getCalledFunction() == F && !SAA.isKernelFunc(CI->getFunction()))
        Calls.push_back(CI);

    for (auto *CI : Calls) {
      HIPSYCL_DEBUG_INFO << "[RemoveBarrierCalls] Replace leftover CBS intrinsic call " << *CI
                         << "\n";
      CI->replaceAllUsesWith(CI->getArgOperand(0));
      CI->eraseFromParent();
      Changed = true;
    }

    if (F->use_empty()) {
      F->eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}

bool removeBarrierCalls(llvm::Function &F, SplitterAnnotationInfo &SAA) {
  if (!SAA.isKernelFunc(&F))
    return false;

  // Collect the barrier calls to be removed first, not remove them
  // instantly as it'd invalidate the iterators.
  llvm::SmallPtrSet<llvm::Instruction *, 8> BarriersToRemove;

  for (auto &BB : F) {
    for (auto &I : BB) {
      if (utils::isBarrier(&I, SAA) || utils::isSubBarrier(&I, SAA)) {
        BarriersToRemove.insert(&I);
      }
    }
  }

  for (auto *B : BarriersToRemove) {
    HIPSYCL_DEBUG_INFO << "[RemoveBarrierCalls] Remove barrier ";
    HIPSYCL_DEBUG_EXECUTE_INFO(B->print(llvm::outs());
                               llvm::outs() << " from " << B->getParent()->getName() << "\n";)
    B->eraseFromParent();
  }

  auto *M = F.getParent();
  if (auto *B = M->getFunction(BarrierIntrinsicName)) {
    if (B->getNumUses() == 0) {
      B->eraseFromParent();
      SAA.removeSplitter(B);
      HIPSYCL_DEBUG_INFO << "[RemoveBarrierCalls] Clean-up helper barrier: "
                         << BarrierIntrinsicName << "\n";
    }
  }
  if (auto *B = M->getFunction(SubBarrierIntrinsicName)) {
    if (B->getNumUses() == 0) {
      B->eraseFromParent();
      SAA.removeSplitter(B);
      HIPSYCL_DEBUG_INFO << "[RemoveBarrierCalls] Clean-up helper sub-barrier: "
                         << SubBarrierIntrinsicName << "\n";
    }
  }

  bool Changed = !BarriersToRemove.empty();

  Changed |= deleteGlobalVariable(M, LocalIdGlobalNameX, SAA);
  Changed |= deleteGlobalVariable(M, LocalIdGlobalNameY, SAA);
  Changed |= deleteGlobalVariable(M, LocalIdGlobalNameZ, SAA);
  Changed |= deleteGlobalVariable(M, SgLocalIdGlobalName, SAA);
  Changed |= deleteGlobalVariable(M, SgIdGlobalName, SAA);
  Changed |= deleteGlobalVariable(M, SgSizeGlobalName, SAA);
  Changed |= deleteGlobalVariable(M, SgNumSubgroupsGlobalName, SAA);
  Changed |= deleteGlobalVariable(M, llvm::StringRef{SubGroupSharedMemory.data(),
                                                     SubGroupSharedMemory.size()}, SAA);
  Changed |= deleteGlobalVariable(M, llvm::StringRef{WorkGroupSharedMemory.data(),
                                                     WorkGroupSharedMemory.size()}, SAA);
  Changed |= removeLeftoverCBSIntrinsics(M, SAA);

  return Changed;
}

} // namespace


char RemoveBarrierCallsPassLegacy::ID = 0;

bool RemoveBarrierCallsPassLegacy::runOnFunction(llvm::Function &F) {
  auto &SAA = getAnalysis<SplitterAnnotationAnalysisLegacy>().getAnnotationInfo();
  return removeBarrierCalls(F, SAA);
}

void RemoveBarrierCallsPassLegacy::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<SplitterAnnotationAnalysisLegacy>();
  AU.addPreserved<SplitterAnnotationAnalysisLegacy>();
}

llvm::PreservedAnalyses RemoveBarrierCallsPass::run(llvm::Function &F,
                                                    llvm::FunctionAnalysisManager &AM) {
  auto &MAM = AM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);
  auto *SAA = MAM.getCachedResult<SplitterAnnotationAnalysis>(*F.getParent());
  if (!SAA)
    return llvm::PreservedAnalyses::all();

  if (!removeBarrierCalls(F, *SAA))
    return llvm::PreservedAnalyses::all();

  llvm::PreservedAnalyses PA;
  PA.preserve<SplitterAnnotationAnalysis>();
  return PA;
}
} // namespace hipsycl::compiler
