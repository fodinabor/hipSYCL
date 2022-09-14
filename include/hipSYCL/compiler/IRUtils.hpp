//
// Created by joachim on 04.06.21.
//

#ifndef HIPSYCL_IRUTILS_HPP
#define HIPSYCL_IRUTILS_HPP

#include "hipSYCL/common/debug.hpp"

#include <llvm/Analysis/LoopInfo.h>

namespace llvm {
class Region;
class AssumptionCache;
} // namespace llvm

namespace hipsycl::compiler {
static constexpr size_t NumArrayElements = 1024;
static constexpr size_t DefaultAlignment = 64;
static constexpr size_t SGSize = 32;
struct MDKind {
  static constexpr const char Arrayified[] = "hipSYCL.arrayified";
  static constexpr const char InnerLoop[] = "hipSYCL.loop.inner";
  static constexpr const char WorkItemLoop[] = "hipSYCL.loop.workitem";
};

static constexpr const char BarrierIntrinsicName[] = "__hipsycl_barrier";
static constexpr const char SubBarrierIntrinsicName[] = "__hipsycl_sg_barrier";
static constexpr const char LocalIdGlobalNameX[] = "_ZN7hipsycl4glue12omp_dispatch20__hipsycl_local_id_xE";
static constexpr const char LocalIdGlobalNameY[] = "_ZN7hipsycl4glue12omp_dispatch20__hipsycl_local_id_yE";
static constexpr const char LocalIdGlobalNameZ[] = "_ZN7hipsycl4glue12omp_dispatch20__hipsycl_local_id_zE";
static const std::array<const char *, 3> LocalIdGlobalNames{LocalIdGlobalNameX, LocalIdGlobalNameY, LocalIdGlobalNameZ};
static constexpr const char SgIdGlobalName[] = "_ZN7hipsycl4sycl15__hipsycl_sg_idE";

class SplitterAnnotationInfo;

namespace utils {
// can be used to make `llvm::SmallPtrSet` compatible with `std::inserter`
template <class PtrSet> struct PtrSetWrapper {
  explicit PtrSetWrapper(PtrSet &PtrSetArg) : Set(PtrSetArg) {}
  PtrSet &Set;
  using iterator = typename PtrSet::iterator;
  using value_type = typename PtrSet::value_type;
  template <class IT, class ValueT> IT insert(IT, const ValueT &Value) { return Set.insert(Value).first; }
};

llvm::Loop *updateDtAndLi(llvm::LoopInfo &LI, llvm::DominatorTree &DT, const llvm::BasicBlock *B, llvm::Function &F);

bool isBarrier(const llvm::Instruction *I, const SplitterAnnotationInfo &SAA);
bool blockHasBarrier(const llvm::BasicBlock *BB, const hipsycl::compiler::SplitterAnnotationInfo &SAA);
bool hasBarriers(const llvm::Function &F, const hipsycl::compiler::SplitterAnnotationInfo &SAA);
bool hasOnlyBarrier(const llvm::BasicBlock *BB, const hipsycl::compiler::SplitterAnnotationInfo &SAA);
bool startsWithBarrier(const llvm::BasicBlock *BB, const hipsycl::compiler::SplitterAnnotationInfo &SAA);
bool endsWithBarrier(const llvm::BasicBlock *BB, const hipsycl::compiler::SplitterAnnotationInfo &SAA);
llvm::CallInst *createBarrier(llvm::Instruction *InsertBefore, hipsycl::compiler::SplitterAnnotationInfo &SAA);

bool isSubBarrier(const llvm::Instruction *I, const hipsycl::compiler::SplitterAnnotationInfo &SAA);
bool blockHasSubBarrier(const llvm::BasicBlock *BB, const hipsycl::compiler::SplitterAnnotationInfo &SAA);
bool hasSubBarriers(const llvm::Function &F, const hipsycl::compiler::SplitterAnnotationInfo &SAA);
bool hasOnlySubBarrier(const llvm::BasicBlock *BB, const hipsycl::compiler::SplitterAnnotationInfo &SAA);
bool startsWithSubBarrier(const llvm::BasicBlock *BB, const hipsycl::compiler::SplitterAnnotationInfo &SAA);
bool endsWithSubBarrier(const llvm::BasicBlock *BB, const hipsycl::compiler::SplitterAnnotationInfo &SAA);
llvm::CallInst *createSubBarrier(llvm::Instruction *InsertBefore, hipsycl::compiler::SplitterAnnotationInfo &SAA);

bool isWorkItemLoop(const llvm::Loop &L);
bool isInWorkItemLoop(const llvm::Loop &L);
bool isInWorkItemLoop(const llvm::Region &R, const llvm::LoopInfo &LI);
/*!
 * Get's the original work item loop.
 * @param LI The LoopInfo used to find the loop.
 * @return The single work item loop annotated with hipSYCL.loop.workitem.
 */
llvm::Loop *getSingleWorkItemLoop(const llvm::LoopInfo &LI);
llvm::BasicBlock *getWorkItemLoopBodyEntry(const llvm::Loop *WILoop);

bool checkedInlineFunction(llvm::CallBase *CI, llvm::StringRef PassPrefix);

bool isAnnotatedParallel(llvm::Loop *TheLoop);

void createParallelAccessesMdOrAddAccessGroup(const llvm::Function *F, llvm::Loop *const &L,
                                              llvm::MDNode *MDAccessGroup);

void addAccessGroupMD(llvm::Instruction *I, llvm::MDNode *MDAccessGroup);

llvm::SmallPtrSet<llvm::BasicBlock *, 8> getBasicBlocksInWorkItemLoops(const llvm::LoopInfo &LI);

llvm::SmallVector<llvm::Loop *, 4> getLoopsInPreorder(const llvm::LoopInfo &LI);

/*!
 * In _too simple_ loops, we might not have a dedicated latch.. so make one!
 * Only simple / canonical loops supported.
 *
 * Also adds vectorization hint to latch, so only use for work item loops..
 *
 * @param L The loop without a dedicated latch.
 * @param Latch The loop latch.
 * @param LI LoopInfo to be updated.
 * @param DT DominatorTree to be updated.
 * @return The new latch block, mostly containing the loop induction instruction.
 */
llvm::BasicBlock *simplifyLatch(const llvm::Loop *L, llvm::BasicBlock *Latch, llvm::LoopInfo &LI,
                                llvm::DominatorTree &DT);

llvm::BasicBlock *splitEdge(llvm::BasicBlock *Root, llvm::BasicBlock *&Target, llvm::LoopInfo *LI = nullptr,
                            llvm::DominatorTree *DT = nullptr);
void promoteAllocas(llvm::BasicBlock *EntryBlock, llvm::DominatorTree &DT, llvm::AssumptionCache &AC);
llvm::Instruction *getBrCmp(const llvm::BasicBlock &BB);

/// Arrayification of work item private values
void arrayifyAllocas(llvm::BasicBlock *EntryBlock, llvm::Loop &L, llvm::Value *Idx, const llvm::DominatorTree &DT);
llvm::AllocaInst *arrayifyValue(llvm::Instruction *IPAllocas, llvm::Value *ToArrayify,
                                llvm::Instruction *InsertionPoint, llvm::Value *Idx,
                                size_t NumValues = hipsycl::compiler::NumArrayElements,
                                llvm::MDTuple *MDAlloca = nullptr);
llvm::AllocaInst *arrayifyInstruction(llvm::Instruction *IPAllocas, llvm::Instruction *ToArrayify, llvm::Value *Idx,
                                      size_t NumValues = hipsycl::compiler::NumArrayElements,
                                      llvm::MDTuple *MDAlloca = nullptr);
llvm::LoadInst *loadFromAlloca(llvm::AllocaInst *Alloca, llvm::Value *Idx, llvm::Instruction *InsertBefore,
                               const llvm::Twine &NamePrefix = "");

llvm::AllocaInst *getLoopStateAllocaForLoad(llvm::LoadInst &LInst);

std::array<size_t, 3> getReqdWgSize(const llvm::Function &F);
size_t getReqdStackElements(const llvm::Function &F);
size_t getReqdSgSize(const llvm::Function &F);

template <class UserType, class Func> bool anyOfUsers(llvm::Value *V, Func &&L) {
  for (auto *U : V->users())
    if (UserType *UT = llvm::dyn_cast<UserType>(U))
      if (L(UT))
        return true;
  return false;
}

template <class UserType, class Func> bool noneOfUsers(llvm::Value *V, Func &&L) {
  return !anyOfUsers<UserType>(V, std::forward<Func>(L));
}

template <class UserType, class Func> bool allOfUsers(llvm::Value *V, Func &&L) {
  return !anyOfUsers<UserType>(V, [L = std::forward<Func>(L)](UserType *UT) { return !L(UT); });
}

/// dbg handling
void copyDgbValues(llvm::Value *From, llvm::Value *To, llvm::Instruction *InsertBefore);
void dropDebugLocation(llvm::Instruction &I);
void dropDebugLocation(llvm::BasicBlock *BB);

/// opaque ptr abstraction
/// we have some cases where originally we had to look through a bitcast / gep
/// but since opaque ptrs, they're gone, so now we check, if \a V is the type we're looking for
template <class T> T *getValueOneLevel(llvm::Constant *V, unsigned idx = 0) {
  // opaque ptr
  if (auto *R = llvm::dyn_cast<T>(V))
    return R;

  // typed ptr -> look through bitcast
  if (V->getNumOperands() == 0)
    return nullptr;
  return llvm::dyn_cast<T>(V->getOperand(idx));
}

} // namespace utils
} // namespace hipsycl::compiler
#endif // HIPSYCL_IRUTILS_HPP
