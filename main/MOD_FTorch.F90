MODULE MOD_FTorch
   USE, INTRINSIC :: iso_fortran_env, ONLY: sp=>real32, dp=>real64
   USE ftorch, ONLY: torch_tensor, torch_model, torch_kCPU, torch_delete, &
                     torch_tensor_from_array, torch_model_load, torch_model_forward
   IMPLICIT NONE

   PUBLIC :: FTorch_init, FTorch_routine, FTorch_final

   INTEGER, PARAMETER :: wp = sp

   TYPE(torch_model), SAVE :: torch_net
   TYPE(torch_tensor), DIMENSION(1), SAVE :: input_tensors
   TYPE(torch_tensor), DIMENSION(1), SAVE :: output_tensors

   ! Fortran (row,col) -> Torch (N, F)
   INTEGER, PARAMETER :: tensor_layout(2) = [1, 2]

   REAL(wp), ALLOCATABLE, TARGET, SAVE :: in_buf(:, :)   ! (nsample, 5)
   REAL(wp), ALLOCATABLE, TARGET, SAVE :: out_buf(:, :)  ! (nsample, 1)

   CHARACTER(len=*), PARAMETER :: model_torchscript_file = &
        '/tera12/yuanhua/dongwz/github/master/CoLM-FTorch/CoLM202X/pytorch/Flaml_hum/lgbm_hb_logtarget_ts.pt'

   LOGICAL, SAVE :: model_loaded = .FALSE.
   INTEGER, SAVE :: last_nsample = -1

CONTAINS

   SUBROUTINE FTorch_init()
      IF (.NOT. model_loaded) THEN
         CALL torch_model_load(torch_net, model_torchscript_file, torch_kCPU)
         model_loaded = .TRUE.
      END IF
   END SUBROUTINE FTorch_init


   SUBROUTINE ensure_bound(nsample)
      INTEGER, INTENT(IN) :: nsample

      IF (.NOT. model_loaded) CALL FTorch_init()

      IF (nsample <= 0) THEN
         STOP 'ensure_bound: nsample must be > 0'
      END IF

      IF (nsample /= last_nsample) THEN
         CALL torch_delete(input_tensors)
         CALL torch_delete(output_tensors)

         IF (ALLOCATED(in_buf))  DEALLOCATE(in_buf)
         IF (ALLOCATED(out_buf)) DEALLOCATE(out_buf)

         ALLOCATE(in_buf(nsample, 5))
         ALLOCATE(out_buf(nsample, 1))

         CALL torch_tensor_from_array(input_tensors(1),  in_buf,  tensor_layout, torch_kCPU)
         CALL torch_tensor_from_array(output_tensors(1), out_buf, tensor_layout, torch_kCPU)

         last_nsample = nsample
      END IF
   END SUBROUTINE ensure_bound


   SUBROUTINE FTorch_routine(in_data, out_data)
      !   in_data  : (nsample, 5)
      !   out_data : (nsample, 1)
      REAL(wp), INTENT(IN),  TARGET :: in_data(:, :)
      REAL(wp), INTENT(OUT), TARGET :: out_data(:, :)

      INTEGER :: nsample, nfeat, nout

      nsample = SIZE(in_data, 1)
      nfeat   = SIZE(in_data, 2)
      IF (nfeat /= 5) STOP 'FTorch_routine: in_data must have 5 features per sample'

      CALL ensure_bound(nsample)

      in_buf(:, :) = in_data(:, :)

      CALL torch_model_forward(torch_net, input_tensors, output_tensors)

      nout = SIZE(out_data, 2)
      IF (nout /= 1 .OR. SIZE(out_data,1) /= nsample) THEN
         STOP 'FTorch_routine: out_data must be (nsample,1) and match nsample'
      END IF
      out_data(:, :) = out_buf(:, :)
   END SUBROUTINE FTorch_routine


   SUBROUTINE FTorch_final()
      CALL torch_delete(input_tensors)
      CALL torch_delete(output_tensors)
      CALL torch_delete(torch_net)

      IF (ALLOCATED(in_buf))  DEALLOCATE(in_buf)
      IF (ALLOCATED(out_buf)) DEALLOCATE(out_buf)

      model_loaded  = .FALSE.
      last_nsample  = -1
   END SUBROUTINE FTorch_final

END MODULE MOD_FTorch


