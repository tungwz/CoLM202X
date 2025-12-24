#include <define.h>
MODULE MOD_Urban_Const_LCZ

! -----------------------------------------------------------------------
! !DESCRIPTION:
!  look-up-table for LCZ morphology and thermal parameters
!                             - NOTE -
!  Each city may have different values for the parameters in this table.
!  The default values may not suit any specific city.
!  Users could adjust these values based on the city they are working with.
!
!  Created by Wenzong Dong, Jun, 2022
!
! !REFERENCES:
!  1) Stewart, I. D., Oke, T. R., & Krayenhoff, E. S. (2014). Evaluation of
!  the 'local climate zone' scheme using temperature observations and model
!  simulations. International Journal of Climatology, 34(4), 1062-1080.
!  https://doi.org/10.1002/joc.3746
!
!  2) The URBPARM_LCZ.TBL of WRF, https://github.com/wrf-model/WRF/
!
! -----------------------------------------------------------------------
! !USE
   USE MOD_Precision

   IMPLICIT NONE
   SAVE

   ! roof fraction [-]
   real(r8), parameter, dimension(10) :: wtroof_lcz &
      = (/0.5 , 0.5 , 0.55, 0.3 , 0.3, 0.3, 0.8 , 0.4 , 0.15, 0.25/)

   ! pervious fraction [-]
   real(r8), parameter, dimension(10)  :: fgper_lcz &
      = (/0.05, 0.1 , 0.15, 0.35, 0.3, 0.4, 0.15, 0.15, 0.7 , 0.45/)

   ! height of roof [m]
   real(r8), parameter, dimension(10)  :: htroof_lcz &
      = (/45., 15. , 5.  , 40., 15., 5. , 3. , 7. , 5.  , 8.5 /)

   ! H/W [-]
   real(r8), parameter, dimension(10)  :: hwrbld_lcz &
      = (/2.5, 1.25, 1.25, 1. , 0.5, 0.5, 1.5, 0.2, 0.15, 0.35/)

   ! thickness of roof [m]
   real(r8), parameter, dimension(10)  :: thkroof_lcz &
      = (/0.3 , 0.3 , 0.2 , 0.3 , 0.25, 0.15, 0.05, 0.12, 0.15, 0.05/)

   ! thickness of wall [m]
   real(r8), parameter, dimension(10)  :: thkwall_lcz &
      = (/0.3 , 0.25, 0.2 , 0.2 , 0.2 , 0.2 , 0.1 , 0.2 , 0.2 , 0.05/)

   ! thickness of impervious road [m]
   real(r8), parameter, dimension(10)  :: thkgimp_lcz &
      = (/0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25/)

   ! albedo of roof [-]
   real(r8), parameter, dimension(10)  :: albroof_lcz &
      = (/0.13, 0.18, 0.15, 0.13, 0.13, 0.13, 0.15, 0.18, 0.13, 0.1 /)

   ! albedo of wall [-]
   real(r8), parameter, dimension(10)  :: albwall_lcz &
      = (/0.25, 0.2 , 0.2 , 0.25, 0.25, 0.25, 0.2 , 0.25, 0.25, 0.2 /)

   ! albedo of impervious road [-]
   real(r8), parameter, dimension(10)  :: albgimp_lcz &
      = (/0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.18, 0.14, 0.14, 0.14/)

   ! albedo of pervious road [-]
   real(r8), parameter, dimension(10)  :: albgper_lcz &
      = (/0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15/)

   ! emissivity of roof [-]
   real(r8), parameter, dimension(10)  :: emroof_lcz &
      = (/0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.28, 0.91, 0.91, 0.91/)

   ! emissivity of wall [-]
   real(r8), parameter, dimension(10)  :: emwall_lcz &
      = (/0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90/)

   ! emissivity of road [-]
   real(r8), parameter, dimension(10)  :: emgimp_lcz &
      = (/0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.92, 0.95, 0.95, 0.95/)

   ! emissivity of impervious road [-]
   real(r8), parameter, dimension(10)  :: emgper_lcz &
      = (/0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95/)

   ! volumetric heat capacity of roof [J/m3*K]
   real(r8), parameter, dimension(10)  :: cvroof_lcz &
      = (/1.8E6 , 1.8E6 , 1.44E6, 1.8E6 , 1.8E6 , 1.44E6, 2.0E6 , 1.8E6 , 1.44E6, 2.0E6 /)

   ! volumetric heat capacity of wall [J/m3*K]
   real(r8), parameter, dimension(10)  :: cvwall_lcz &
      = (/1.8E6 , 2.67E6, 2.05E6, 2.0E6 , 2.0E6 , 2.05E6, 0.72E6, 1.8E6 , 2.56E6, 1.69E6/)

   ! volumetric heat capacity of impervious road [J/m3*K]
   real(r8), parameter, dimension(10)  :: cvgimp_lcz &
      = (/1.75E6, 1.68E6, 1.63E6, 1.54E6, 1.50E6, 1.47E6, 1.67E6, 1.38E6, 1.37E6, 1.49E6/)

   ! thermal conductivity of roof [W/m*K]
   real(r8), parameter, dimension(10)  :: tkroof_lcz &
      = (/1.25, 1.25, 1.00, 1.25, 1.25, 1.00, 2.0 , 1.25, 1.00, 2.00/)

   ! thermal conductivity of wall [W/m*K]
   real(r8), parameter, dimension(10)  :: tkwall_lcz &
      = (/1.09, 1.5 , 1.25, 1.45, 1.45, 1.25, 0.5 , 1.25, 1.00, 1.33/)

   ! thermal conductivity of impervious road [W/m*K]
   real(r8), parameter, dimension(10)  :: tkgimp_lcz &
      = (/0.77, 0.73, 0.69, 0.64, 0.62, 0.60, 0.72, 0.51, 0.55, 0.61/)

   !TODO:AHE coding
   ! maximum temperature of inner room [K]
   real(r8), parameter, dimension(10)  :: tbldmax_lcz &
      = (/297.65, 297.65, 297.65, 297.65, 297.65, 297.65, 297.65, 297.65, 297.65, 297.65/)

   ! minimum temperature of inner room [K]
   real(r8), parameter, dimension(10)  :: tbldmin_lcz &
      = (/290.65, 290.65, 290.65, 290.65, 290.65, 290.65, 290.65, 290.65, 290.65, 290.65/)

!   real(r8), parameter, dimension(10)  :: tbldmin_lcz &
!      = (/288.15, 288.15, 288.15, 288.15, 288.15, 288.15, 288.15, 288.15, 288.15, 288.15/)

   real(r8), parameter, dimension(10)  :: hequip &
      = 0.!(/1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5/)! 14./)

   real(r8), parameter, dimension(24)  :: Equ_scator &
      = (/0.76, 0.72, 0.71, 0.71, 0.72, 0.72, 0.76, 0.80, 0.86, 0.90, 0.91, 0.92, &
          0.91, 0.93, 0.93, 0.93, 0.96, 0.99, 1.00, 0.98, 0.94, 0.90, 0.85, 0.81/)
! 2020
!   real(r8), parameter, dimension(21)  :: scal_f &
!      = (/0.230, 0.244, 0.265, 0.279, 0.298, 0.328, 0.354, 0.399, 0.429, 0.474, &
!          0.510, 0.560, 0.611, 0.660, 0.715, 0.784, 0.852, 0.906, 0.973, 1.028, 1./)

!2015
!   real(r8), parameter, dimension(21)  :: scal_f &
!      = (/0.293, 0.311, 0.337, 0.356, 0.380, 0.418, 0.452, 0.509, 0.547, 0.604, &
!          0.651, 0.714, 0.779, 0.842, 0.912, 1.000, 1.090, 1.160, 1.240, 1.310, 1.280/)

!2010
!    real(r8), parameter, dimension(21)  :: scal_f &
!       = (/0.637, 0.664, 0.670, 0.689, 0.689, 0.681, 0.722, 0.793, 0.867, 0.882, 1., &
!           1.083,    1.,    1.,    1.,    1.,    1.,    1.,    1.,    1.,     1. /)

!tot_GDP/tot_urb, GlobPOP, 2019
!    real(r8), parameter, dimension(21)  :: scal_f &
!       = (/0.42639472_r8, 0.45072035_r8, 0.45902860_r8, 0.44075042_r8, 0.44548750_r8, &
!           0.46723607_r8, 0.49686128_r8, 0.57134899_r8, 0.65635625_r8, 0.67718912_r8, &
!           0.71975997_r8, 0.76632695_r8, 0.77519995_r8, 0.79007075_r8, 0.83375496_r8, &
!           0.84143918_r8, 0.85961606_r8, 0.91375684_r8, 0.95496166_r8, 1.00000000_r8, &
!           1.03607862_r8/)

!tot_GDP/tot_EC, 2020
!    real(r8), parameter, dimension(21)  :: scal_f &
!       =(/0.33055007, 0.34499629, 0.34937917, 0.34114338, 0.34522469, &
!          0.35018427, 0.37639623, 0.42821663, 0.49394640, 0.51105029, &
!          0.56182123, 0.61588769, 0.65926743, 0.69711075, 0.74397082, &
!          0.78172035, 0.83270809, 0.90106915, 0.95844608, 0.99866887, &
!          1.00000000 /)

!tot_GDP/tot_urb, GlobPOP, 2019
!    real(r8), parameter, dimension(21)  :: scal_f &
!       = (/0.42639472_r8, 0.45072035_r8, 0.45902860_r8, 0.44075042_r8, 0.44548750_r8, &
!           0.46723607_r8, 0.49686128_r8, 0.57134899_r8, 0.65635625_r8, 0.67718912_r8, &
!           0.71975997_r8, 0.76632695_r8, 0.77519995_r8, 0.79007075_r8, 0.83375496_r8, &
!           0.84143918_r8, 0.85961606_r8, 0.91375684_r8, 0.95496166_r8, 1.00000000_r8, &
!           1.03607862_r8/)

!tot_GDP/tot_urb, GlobPOP, 2020
!    real(r8), parameter, dimension(21)  :: scal_f &
!       = (/0.41154668, 0.43502524, 0.44304417, 0.42540248, 0.42997461, &
!           0.45096585, 0.47955944, 0.55145331, 0.63350043, 0.65360785, &
!           0.69469629, 0.73964169, 0.74820572, 0.76255868, 0.80472171, &
!           0.81213835, 0.82968226, 0.88193774, 0.92170771, 0.96517772, &
!           1.00000000 /)

!tot_GDP/tot_urb, LandScan, 2020
    real(r8), parameter, dimension(21)  :: scal_f &
       = (/0.41218280, 0.44571636, 0.45103396, 0.42811278, 0.43161228, &
           0.45301755, 0.48416780, 0.55459915, 0.62888020, 0.64964293, &
           0.69982022, 0.74259214, 0.74763603, 0.76126272, 0.80513310, &
           0.80443265, 0.82888435, 0.88229239, 0.92333555, 0.96604183, &
           1.00000000 /)

    real(r8), parameter, dimension(10) :: cap_heat &
       = (/300., 75., 75., 50., 25., 25., 35., 50., 10., 300./)
END MODULE MOD_Urban_Const_LCZ
! ---------- EOP ------------
