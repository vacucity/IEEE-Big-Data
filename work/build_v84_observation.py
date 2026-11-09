"""Use the validated observation-only residual with the common V84 auditor."""
import build_v84_context as B
import t1_v84_context_continuation as J

if __name__=='__main__':
    B.K=J;B.OUT=J.OUT;B.FEATURE_OUT=J.DATA_OUT;B.NAME='v84_observation_flow'
    B.build()
