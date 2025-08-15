MODULE MOD_FTorch

   USE, intrinsic :: iso_fortran_env, only : sp=>real32

   USE ftorch, only : torch_tensor, torch_model, torch_kCPU, torch_delete, &
                      torch_tensor_from_array, torch_model_load, torch_model_forward


   IMPLICIT NONE

   PUBLIC FTorch_init, FTorch_routine, FTorch_final

   type(torch_model ) :: torch_net
   type(torch_tensor), dimension(1) :: input_tensors
   type(torch_tensor), dimension(1) :: output_tensors

   INTEGER, parameter :: wp = sp
   INTEGER, parameter, dimension(2) :: tensor_layout = [1, 2]

   real(wp), target, save  :: in_buf(1,10)
   real(wp), target, save  :: out_buf(1,1)

   CHARACTER(len=128) :: model_torchscript_file

   CONTAINS

   SUBROUTINE FTorch_init()

      ! Load ML model
      ! model_torchscript_file = '/tera12/yuanhua/dongwz/soft/FTorch/examples/5_Looping/saved_simplenet_cpu.pt'
      model_torchscript_file = '/tera12/yuanhua/dongwz/github/master/CoLM-FTorch/CoLM202X/pytorch/RF/rf_regressor_ts.pt'
      CALL torch_model_load(torch_net, model_torchscript_file, torch_kCPU)

      CALL torch_tensor_from_array(input_tensors(1), in_buf, tensor_layout, torch_kCPU)
      CALL torch_tensor_from_array(output_tensors(1), out_buf, tensor_layout, torch_kCPU)

   END SUBROUTINE FTorch_init


   SUBROUTINE FTorch_routine(in_data, out_data)

      ! Set up Fortran data structures
      REAL(wp), dimension(1, 10), target, intent(IN)  :: in_data
      REAL(wp), dimension(1,  1), target, intent(OUT) :: out_data

      ! Create Torch input/output tensors from the above arrays
      ! CALL torch_tensor_from_array(input_tensors(1), in_data, tensor_layout, torch_kCPU)
      ! CALL torch_tensor_from_array(output_tensors(1), out_data, tensor_layout, torch_kCPU)

      in_buf  = in_data
      ! Infer
      CALL torch_model_forward(torch_net, input_tensors, output_tensors)
      out_data = out_buf

   END SUBROUTINE FTorch_routine

   SUBROUTINE FTorch_final()
      ! Cleanup
      CALL torch_delete(input_tensors)
      CALL torch_delete(output_tensors)
      CALL torch_delete(torch_net)

   END SUBROUTINE FTorch_final
END MODULE MOD_FTorch
