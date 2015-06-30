! -*- f90 -*-
! geotecha - A software suite for geotechncial engineering
! Copyright (C) 2015  Rohan T. Walker (rtrwalker@gmail.com)
!
! This program is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
!
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License
! along with this program.  If not, see http://www.gnu.org/licenses/gpl.html.
!
!
!Elasto-Plastic for Unsaturated Soils, adapted from VB code in
!Phd thesis of Pham (2005)
!
!The code is largely the same as that of Pham (2005).  However, some changes
!were neccesary due to f2py inability to use derived types.  Hence,
!rather than say 'ss' being an attribute of a 'DataResult' type, (i.e.
!datapoint%ss) it is merely defined as 'ss' as a public variable in the
!module name space.  Som minor name changes were needed to prevent
!name conflict.
!
!Comment like "!'A-80" refer to pages in Appendix A of Pham (2005).
!Basically the code loosely follows the Pham form.
!
!Remember if using this through python f2py that all variables and
!function names are LOWER CASE!
!
!For how to use see the 'main' program at the end of the file.
!
! compile to an exe with: gfortran epus_ext.f95 -o epus.exe
!References
!----------
!Pham, H. Q., and Fredlund, D. G. (2011).
!       'Volume-mass unsaturated soil constitutive model for drying-wetting
!       under isotropic loading-unloading conditions. Canadian Geotechnical
!       Journal, 48(2), 280-313.
!Pham, H. Q. (2005). 'A volume-mass constitutive model for
!       unsaturated soils'. PhD Thesis, University of Saskatchewan,
!       Saskatoon, Saskatchewan, Canada.


      MODULE types
      ! 'types' module defines double precision
        IMPLICIT NONE
        INTEGER, PARAMETER :: DP = KIND(0.0D0)

      END MODULE

      MODULE parameters
      ! 'parameters' module defines some double precision constants
        USE types
        IMPLICIT NONE
        REAL(KIND=DP), PARAMETER :: zero = 0.0_DP
        REAL(KIND=DP), PARAMETER :: one = 1.0_DP
        REAL(KIND=DP), PARAMETER :: two = 2.0_DP
        REAL(KIND=DP), PARAMETER :: three = 3.0_DP
        REAL(KIND=DP), PARAMETER :: six = 6.0_DP
        REAL(KIND=DP), PARAMETER :: ten = 10.0_DP
      END MODULE parameters

      MODULE epus
      ! 'epus' module implements Pham(2005) Elasto-Plastic for Unsaturated Soils
        USE types
        USE parameters
        IMPLICIT NONE

!         TYPE CurveFittingSWCC
          ! Soil water characteristic curve
          REAL(DP) :: wsat ! Gravimetric water condent at 100% saturation.
          REAL(DP) :: a ! Curve fitting parameter.
          REAL(DP) :: b ! Curve fitting parameter.
          REAL(DP) :: wr ! Residual gravimetric water content.
          REAL(DP) :: sl ! Initial slope of SWCC.
!         END TYPE


!         TYPE PoresizeDistribution  ! Pore size distribution curve

          INTEGER :: Npoint         ! Number of data points along the pore size distribution curve
          REAL(DP), ALLOCATABLE, DIMENSION(:) :: AEV ! Value of Air Entry Value that the function F represents
          REAL(DP), ALLOCATABLE, DIMENSION(:) :: WEV ! Value of Water Entry Value that the function F represents
          REAL(DP), ALLOCATABLE, DIMENSION(:) :: Ccp ! Virgin compression index of the group of pore at saturated
          REAL(DP), ALLOCATABLE, DIMENSION(:) :: Csp ! Unloading-Reloading index of the group of pore at saturated
          REAL(DP), ALLOCATABLE, DIMENSION(:) :: Ccdp ! Virgin compression index of the group of pore at completely dry condition (10**6 kPa)
          REAL(DP), ALLOCATABLE, DIMENSION(:) :: RV ! Ratio of Volume of the group of pore/ volume of solid phase.
          REAL(DP), ALLOCATABLE, DIMENSION(:) :: YieldSt ! Equivalent maximum stress acted on the pores at water-filled state
          LOGICAL, ALLOCATABLE, DIMENSION(:) :: Filled ! State of the pore - either filled or empty.
          REAL(DP), ALLOCATABLE, DIMENSION(:) :: RVC ! Actual ratio of volume of the group of pore/ volume of solid phase
          REAL(DP), ALLOCATABLE, DIMENSION(:) :: AEVC ! Actual Air Entry Value of the group of pores (i) which has AEV at dried slurry = AEV(i)
          REAL(DP), ALLOCATABLE, DIMENSION(:) :: WEVC ! Actual water entry value
          LOGICAL, ALLOCATABLE, DIMENSION(:) :: Airentrapped ! If experienced dried state = true, if did not = false
!         END TYPE



!         TYPE StressPath                  !Describe all stress paths
          INTEGER :: stpn ! counter
          INTEGER :: nsteps ! number of stress path steps
          LOGICAL, ALLOCATABLE, DIMENSION(:) :: ist ! If change in stress...ist=1 else ist=0
          REAL(DP), ALLOCATABLE, DIMENSION(:) :: vl ! value of suction or stress
          INTEGER, ALLOCATABLE, DIMENSION(:) :: npp ! Number of points to subdivide stress path step into.
          INTEGER, ALLOCATABLE, DIMENSION(:) :: startpoint ! Describe starting for a stress path
          INTEGER, ALLOCATABLE, DIMENSION(:) :: endpoint ! Describe ending point for a stress path
!         END TYPE

!         TYPE DataResults
          INTEGER :: datapointn ! counter
          INTEGER :: npts ! totoal number of data points
          REAL(DP), ALLOCATABLE, DIMENSION(:) :: ss ! suction
          REAL(DP), ALLOCATABLE, DIMENSION(:) :: st ! net normal stress
          REAL(DP), ALLOCATABLE, DIMENSION(:) :: ee ! void ratio
          REAL(DP), ALLOCATABLE, DIMENSION(:) :: w ! gravimetric water content
          REAL(DP), ALLOCATABLE, DIMENSION(:) :: Sr ! degree of saturation
          REAL(DP), ALLOCATABLE, DIMENSION(:) :: vw ! volumnetric water content
!         END TYPE



!         TYPE(CurveFittingSWCC) :: SimpleSWCC ! Type of SWCC equation that will be used in the development of the model.
!         TYPE(PoresizeDistribution) :: f ! Pore size distribution function at initial condition.
!         TYPE(StressPath) :: stp !Stress path along with how many point along each stress path
        CHARACTER(LEN=100) :: defaultdir
        CHARACTER(LEN=100) :: Datafilename
        CHARACTER(LEN=100) :: soilname
        CHARACTER(LEN=100) :: username
        REAL(DP) :: Ccs ! semi-log slope of saturated compression curve
        REAL(DP) :: Css ! semi-log slope of saturated recompression curve
        REAL(DP) :: Ccd ! semi-log slope of dry compression curve
        REAL(DP) :: Gs ! Specific gravity of solids
        REAL(DP) :: MinSuction ! log10 of minimum suction, usually -3.
        REAL(DP) :: MaxSuction ! log10 of maximum suction, usually 6.
        REAL(DP) :: Interval ! interval lenght between pore sizes
        REAL(DP) :: Suction ! Current stress
        REAL(DP) :: Stress ! Current soil suction
!         INTEGER :: LoadCond ! = 0 for isotropic; = 1 for Ko
        REAL(DP) :: LogDS ! Distance ratio between boundary drying and boundary drying curve. In log cycles.
!'A-80
        REAL(DP) :: LogRS ! Slope ratio between boundary drying and boundary drying curve. In log cycles
!         TYPE(DataResults) :: datapoint
!        INTEGER :: ssvar
!        INTEGER :: prvar
!        INTEGER :: plstp
        REAL(DP) :: beta ! air entrapped in a collapsible pore is proportional to the volume of the pore and can be denoted by an entrapped air parameter beta.
        REAL(DP) :: K0 ! Lateral earth pressure coefficient. Use 1 for isotropic
        REAL(DP), ALLOCATABLE, DIMENSION(:) :: Srdry ! Drying degree of saturation SWCC (zero net mean stress)
        REAL(DP), ALLOCATABLE, DIMENSION(:) :: Srwet !' Wetting degree of saturation SWCC  (zero net mean stress)
        INTEGER :: NumSr ! Number points divided along the wetting and drying Sr
        REAL(DP) :: RefSr ! Reference degree of saturation
        INTEGER :: Curveon ! = 1: on drying; = 2: on horizontal scanning; = 3: on wetting
        INTEGER :: Assumption ! = 0: dry pores are incompressible; =1: dry pores are compressible
        REAL(DP) :: pm ! Soil parameter (Sr)**pm for calculating the yield stress of dry pores
        REAL(DP) :: Pore_shape ! For describing the change in AEV/WEV of the soil
        LOGICAL :: errocc

        LOGICAL :: DEBUG ! for possible dubugging



        CONTAINS
          SUBROUTINE initialize_stp ()!(nsteps_)
          ! Allocate any variables associated with the stresspath
!             USE types
!             USE parameters
            IMPLICIT NONE
!             INTEGER, intent(in) :: nsteps_


!             stpn = nsteps_
!             nsteps = nsteps_
!             ALLOCATE(ist(1:nsteps_))
!             ALLOCATE(vl(1:nsteps_))
!             ALLOCATE(npp(1:nsteps_))
            ALLOCATE(startpoint(1:nsteps))
            ALLOCATE(endpoint(1:nsteps))

          END SUBROUTINE initialize_stp

          SUBROUTINE initialize_f()
          ! Allocate any variables associated with the pore size distrbution
!             USE types
!             USE parameters
            IMPLICIT NONE
!             INTEGER, intent(in) :: Npoint_

!             Npoint = Npoint_
            ALLOCATE(AEV(1:Npoint)) ! Value of Air Entry Value that the function F represents.
            ALLOCATE(WEV(1:Npoint)) ! Value of Water Entry Value that the function F represents.
            ALLOCATE(Ccp(1:Npoint)) ! Virgin compression index of the group of pore at saturated.
            ALLOCATE(Csp(1:Npoint)) ! Unloading-Reloading index of the group of pore at saturated.
            ALLOCATE(Ccdp(1:Npoint)) ! Virgin compression index of the group of pore at completely dry condition (10^6 kPa).
            ALLOCATE(RV(1:Npoint)) ! Ratio of Volume of the group of pore/ volume of solid phase.
            ALLOCATE(YieldSt(1:Npoint)) ! Equivalent maximum stress acted on the pores at water-filled state.
            ALLOCATE(Filled(1:Npoint)) ! State of the pore - either filled or empty.
            ALLOCATE(RVC(1:Npoint)) ! Actual ratio of volume of the group of pore/ volume of solid phase.
            ALLOCATE(AEVC(1:Npoint)) ! Actual Air Entry Value of the group of pores [i] which has AEV at dried slurry = AEV[i].
            ALLOCATE(WEVC(1:Npoint)) ! Actual water entry value.
            ALLOCATE(Airentrapped(1:Npoint)) ! If experienced dried state = True, if did not = False.
          END SUBROUTINE initialize_f

          SUBROUTINE initialize_SrDry_SrWet
          ! Allocate any variables associated with wet and dry saturation curves
            USE types
            USE parameters
            IMPLICIT NONE
            !NumSr is already defined somewhere else


            ALLOCATE(SrDry(1:NumSr))
            ALLOCATE(SrWet(1:NumSr))
          END SUBROUTINE initialize_SrDry_SrWet

         SUBROUTINE initialize_datapoint()!(npts_)
         ! Allocate any variables associated with the results
!             USE types
!             USE parameters
            IMPLICIT NONE
            INTEGER :: i
!             INTEGER, intent(in) :: npts_

!             npts = npts_
            npts = 0
            DO i = 1, nsteps
              npts = npts + npp(i)
            END DO
            ALLOCATE(ss(1:npts))
            ALLOCATE(st(1:npts))
            ALLOCATE(ee(1:npts))
            ALLOCATE(w(1:npts))
            ALLOCATE(Sr(1:npts))
            ALLOCATE(vw(1:npts))

          END SUBROUTINE initialize_datapoint



          FUNCTION CVStress() !This function is used to convert (K0 stress or Isotropic stress) to Isotropic stress
            USE types
            USE parameters
            IMPLICIT NONE
            REAL(DP) :: CVStress

!             IF (LoadCond .EQ.0) THEN
  !'A-82
!               CVStress = one
!             ELSE
              CVStress = (two * K0 + one) / three
!             END IF
          END FUNCTION CVStress

          FUNCTION Wc(s) !Collapsible water content function
            USE types
            USE parameters
            IMPLICIT NONE
            REAL(DP) :: Wc, s
            REAL(DP) :: b1, a1, wsat1, wr1, sl1

            wsat1 = wsat
            wr1 = wr
            b1 = b
            a1 = a
            sl1 = sl
            Wc = (wsat1 - sl1 * Log10(s) / Gs - wr1) * a1 &
                * (one - s ** b1 / ten ** (six * b1)) / (s ** b1 + a1)
          END FUNCTION Wc

          FUNCTION DWc(s) !Differentiation of function Wc at suction s.
            USE types
            USE parameters
            IMPLICIT NONE
            REAL(DP) :: DWc, s
            REAL(DP) :: temp1, temp2, temp3, temp4, temp5, temp6
            REAL(DP) :: b1, a1, wsat1, wr1, sl1 !Temp. variables

            wsat1 = wsat
            wr1 = wr
            b1 = b
            a1 = a
            sl1 = sl
            temp1 = wsat1 - sl1 * Log(s) / (Log(ten) * Gs) - wr1
            temp2 = -a1 * b1 * s ** b1 * (ten ** (-six * b1) * a1 + one) &
                / (s * (s ** b1 + a1) ** two)
            temp3 = (a1 - a1 * s ** b1 / (ten ** (six * b1))) &
                / (s ** b1 + a1)
            temp4 = sl1 / (s * Log(ten) * Gs)
            temp5 = one - (Log(one + s / a1 ** (one / b1)) &
                / Log(one + ten ** six / a1 ** (one / b1)))
            temp6 = a1 ** (one / b1) * (one + s / a1 ** (one / b1)) &
                * Log(one + ten ** six / a1 ** (one / b1))
            DWc = (temp1 * temp2 - temp3 * temp4) * temp5 &
                - (temp3 * temp1 + wr1) / temp6
          END FUNCTION DWc

          SUBROUTINE DryPoreSize
            USE types
            USE parameters
            IMPLICIT NONE
            INTEGER :: i, j
            REAL(DP) :: temp, Vtotal, saturated
            REAL(DP) :: s ! Soil Suction
            REAL(DP) :: b1, a1, wsat1, wr1, sl1, bwt, dwt ! Temp. variables

            wsat1 = wsat
            wr1 = wr
            b1 = b
            a1 = a
            sl1 = sl
            !--------------------------------------------------------------------------------------
  !           NumSr = 400
  !           MaxSuction = 6                 '(Log Scale or 1000000)
  !           MinSuction = -3                '(Log scale or 0.001)
  !           Npoint = 1000                ' Number of points along the pore size distribution curve
  !         '=======================================================================================
  !           ' Total volume at completely dry
  !           Vtotal = 0.0
  !           DO i = 1, Npoint
  !
  !             Interval = (MaxSuction - MinSuction) / Npoint
  !             s = ten ** (AEV(i))
  !             temp = (a1 - a1 * s ** b1 / (ten ** (six * b1))) &
  !                 / (s ** b1 + a1)
  !             Vtotal = Vtotal + (-Gs * DWc(s) * s * Log(ten) &
  !                 - temp * Ccs) * Interval
  !           END DO

            DO i = 1, Npoint
              Interval = (MaxSuction - MinSuction) / Npoint
              ! Air Entry Value
              AEV(i) = (i - 1) * Interval + MinSuction
              s = ten ** (AEV(i))
              !-------------------------------------------------------------------------------
              ! Pore volume
              temp = (a1 - a1 * s ** b1 / (ten ** (six * b1))) &
                  / (s ** b1 + a1)
              RV(i) = (-Gs * DWc(s) * s * Log(ten) - temp * Ccs) &
                  * Interval
              !-------------------------------------------------------------------------------
              ! Saturated compression indexes
              temp = -a1 * b1 * (s ** b1) * (a1 * ten ** (-six * b1) &
                  + one) / (s * (s ** b1 + a1) ** two)
          !A-83

              Ccp(i) = -temp * Ccs * (ten ** (AEV(i) + Interval) &
                  - ten ** (AEV(i)))
              Csp(i) = -temp * Css * (ten ** (AEV(i) + Interval) &
                  - ten ** (AEV(i)))
              ! Dried compression index
              Ccdp(i) = Ccp(i) * (Ccd / Ccs)  !(RV(i) / Vtotal) * Ccd
              !-------------------------------------------------------------------------------
              ! Calculate wetting curve parameters
              bwt = (a1 / (ten ** LogDS) ** b1) ** (one / LogRS)
              dwt = b1 / LogRS
              !-------------------------------------------------------------------------------
              ! Calculate wetting soil suctions (assuming tails of the SWCC are linear)
              IF (AEV(i) .GT. Log10((2.7_DP * a1) ** (one / b1))) Then
                WEV(i) = AEV(i) &
                    - ((Log10((2.7_DP * a1) ** (one / b1)) &
                    - Log10((2.7_DP * bwt) ** (one / dwt))) &
                    * (six - AEV(i)) / (six &
                    - Log10((2.7_DP * a1) ** (one / b1))))
              Else
                WEV(i) = (one / dwt) * (Log10(bwt) + b1 * AEV(i) &
                  - Log10(a1))
              END IF
              RVC(i) = RV(i)
            END DO
          END SUBROUTINE DryPoreSize

          SUBROUTINE SlurryPoreSize ! Zero soil suction - Mean kPa net mean stress.
            USE types
            USE parameters
            IMPLICIT NONE
            INTEGER :: i, j
            REAL(DP) :: temp
            REAL(DP) :: s !soil suction
            REAL(DP) :: b1, a1, wsat1, wr1, sl1 ! Temp. variables

            CALL DryPoreSize()

            DO i = 1, Npoint
              RV(i) = RV(i) + (AEV(i) - MinSuction) * Ccp(i)
              RVC(i) = RV(i)
              AEVC(i) = AEV(i)
              WEVC(i) = WEV(i)
              YieldSt(i) = 0.0001
              Filled(i) = .True.
              Airentrapped(i) = .False.
            END DO

            temp = 0
            DO i = 1, Npoint
              temp = temp + RV(i)
            END DO
            ! Set current suction and stress to the slurry condition.
            Suction = ten ** MinSuction           ! Max suction
            Stress = (ten ** MinSuction) / 1000.0_DP   ! Min suction
            Curveon = 1                         ! Starts from drying curve
            RefSr = 1                           ! Initial degree of saturation = 1
          END SUBROUTINE SlurryPoreSize



          Function ChangeSuction(MaxSumStress, Initialsuction) ! = suction(0)/suction(p)
            USE types
            USE parameters
            IMPLICIT NONE
            REAL(DP) :: MaxSumStress, Initialsuction, ChangeSuction
            REAL(DP) :: temp1, temp2, tmp, ChangeSuction1

              ! Set history parameter equal to 1 for all cases
              IF (Initialsuction >= one * MaxSumStress) THEN
                ChangeSuction = one
                RETURN !RETURN
              END IF
              !Check this condition... if it is collapsible pore group
              IF (Initialsuction >= (ten * a) &
                  ** (one / b)) THEN
                ChangeSuction = one
                RETURN ! RETURN
              END IF
              temp2 = (wsat * Gs - Ccs &
                  * Log10(Initialsuction) - wr * Gs)
              IF (temp2 <= zero) THEN
                ChangeSuction = one
                RETURN! RETURN
              END IF
              temp1 = (Ccs - Css) * Log10(MaxSumStress) + Css &
                  * Log10(Initialsuction + Stress) &
                  - Ccs * Log10(Initialsuction)
              IF ((one - Pore_shape * (temp1 / &
                  (three * temp2))) <= zero) THEN
          !' A-84
                errocc = .true.
                ChangeSuction = one
              Else
                ChangeSuction = (one - Pore_shape &
                    * (temp1 / (three * temp2)))
             END IF
          End Function


        FUNCTION ChangeWetSuction(MaxSumStress, &
                                  Airentryvalue, &
                                  Waterentryvalue) ! = suction(0)/suction(p)
          USE types
          USE parameters
          IMPLICIT NONE
          REAL(DP) :: MaxSumStress, Airentryvalue, Waterentryvalue
          REAL(DP) :: ChangeWetSuction
          REAL(DP) :: temp1, temp2, tmp

          ! Set history parameter equal to 1 for all cases
          IF (Airentryvalue >= one * MaxSumStress) THEN
            ChangeWetSuction = one
            RETURN
          END IF
          !Check this condition...  if it is collapsible pore group
          IF (Airentryvalue >= (ten * a) &
              ** (one / b)) Then
            ChangeWetSuction = one
            RETURN
          END IF
          temp2 = (wsat * Gs - Ccs * Log10(Airentryvalue) &
              - wr * Gs)
          IF (temp2 < zero) THEN
            ChangeWetSuction = one
            RETURN
          END IF
          temp1 = (Ccs - Css) * Log10(MaxSumStress) &
              + Css * Log10(Waterentryvalue + Stress) &
              - Ccs * Log10(Airentryvalue)
          IF ((one - Pore_shape * (temp1 &
              / (three * temp2))) <= zero) THEN
            errocc = .True.
            ChangeWetSuction = one
          Else
            ChangeWetSuction = (one - Pore_shape &
                * (temp1 / (three * temp2)))
          END IF
        END FUNCTION ChangeWetSuction


        Function Changevolume(InitialVolume, &
                              Yieldstress, &
                              CurrentStress, &
                              CompIndex, &
                              UnloadIndex)
          USE types
          USE parameters
          IMPLICIT NONE
          REAL(DP) :: InitialVolume, Yieldstress, CurrentStress
          REAL(DP) :: CompIndex, UnloadIndex, Changevolume
          REAL(DP) :: temp, temp2

          temp = InitialVolume - (Log10(Yieldstress) - MinSuction) &
              * CompIndex + (Log10(Yieldstress) &
              - Log10(CurrentStress)) * UnloadIndex
          If (temp > zero) Then
            Changevolume = temp
          Else
            Changevolume = zero
          End If
        End Function Changevolume

        SUBROUTINE Drying(ss_new)       !'Increase soil suction at a certain net mean/vertical stress
                                        !'If increase soil suction at a certain net mean stress. All pores that has air
          USE types
          USE parameters
          IMPLICIT NONE
          REAL(DP), intent(in) :: ss_new
          INTEGER :: i
          REAL(DP) :: s, sa, ys             !s= real suction

          Select Case (Curveon)
            Case (1)
               Curveon = 1
               RefSr = RfSr_value(1, ss_new)
            Case (2)
               If (RefSr <= RfSr_value(1, ss_new)) Then
                 Curveon = 2
               Else
                 Curveon = 1
                 RefSr = RfSr_value(1, ss_new)
               End If
            Case (3)
               If (RefSr <= RfSr_value(1, ss_new)) Then
                 Curveon = 2
               Else
                 Curveon = 1
                 RefSr = RfSr_value(1, ss_new)
               End If
        !'A-85
          End Select
          DO i = 1, Npoint
            s = ten ** AEV(i)
            If (Stress + s > YieldSt(i)) Then
              ys = Stress + s
            Else
              ys = YieldSt(i)
            End If
            If (Filled(i) .EQV. .True.) Then
              If (ChangeSuction(YieldSt(i), s) .EQ. zero) Then
                !MsgBox "ssdsdd"
                CONTINUE
             End If
              If ((s / ChangeSuction(YieldSt(i), s)) <= ss_new) Then
                If ((Stress + (s / ChangeSuction(ys, s))) &
                    > YieldSt(i)) Then
                  YieldSt(i) = (s / ChangeSuction(ys, s)) + Stress
                End If
                RVC(i) = Changevolume(RV(i), YieldSt(i), &
                    (s / ChangeSuction(ys, s)) + Stress, &
                    Ccp(i), Csp(i))
                Filled(i) = .False.
                Airentrapped(i) = .True.
              Else
                !For pores that have AEV>ss_new - filled with water and subject to the same stress
                If (Stress + ss_new > YieldSt(i)) Then
                  YieldSt(i) = Stress + ss_new
                End If
                RVC(i) = Changevolume(RV(i), YieldSt(i), &
                    Stress + ss_new, Ccp(i), Csp(i))
              End If
            Else
              ! Do not care about the pore that is already dried
              CONTINUE
            End If
          END DO
          Suction = ss_new     ! Change current suction to the new soil suction.
        END SUBROUTINE Drying

        SUBROUTINE Wetting(ss_new)      !'Decrease soil suction at a certain net mean/vertical stress
          USE types
          USE parameters
          IMPLICIT NONE
          REAL(DP), intent(in) :: ss_new
          INTEGER :: i
          REAL(DP) :: s, sa, ys, tmp, ys1      !s= real suction

         Select Case (Curveon)
            Case (1)
               If (RefSr >= RfSr_value(3, ss_new)) Then
                 Curveon = 2
               Else
                 Curveon = 3
                 RefSr = RfSr_value(3, ss_new)
               End If
            Case (2)
               If (RefSr >= RfSr_value(3, ss_new)) Then
                 Curveon = 2
               Else
                 Curveon = 3
                 RefSr = RfSr_value(3, ss_new)
               End If
            Case (3)
               Curveon = 3
               RefSr = RfSr_value(3, ss_new)
          End Select
          DO i = 1, Npoint
            s = ten ** WEV(i)
            If (.Not. Filled(i)) Then     ! Check if the pore group is filled with water
              If (Stress + s > YieldSt(i)) Then
                ys = Stress + s
              Else
                ys = YieldSt(i)
              End If
              If ((s / ChangeWetSuction(ys, ten ** AEV(i), &
                  ten ** WEV(i))) >= ss_new) Then      ! Check if the pore group is possible to be filled with water after suction decreased to ss_new
                If ((s / ChangeWetSuction(ys, ten ** AEV(i), &
                    ten ** WEV(i))) + Stress > YieldSt(i)) Then
                  !'MsgBox "never happend:" + Str(YieldSt(i)) + "     <  " + Str((s / ChangeWetSuction(ys, ten ** AEV(i), ten ** WEV(i))) + Stress)
                  YieldSt(i) = (s / ChangeWetSuction(ys, &
                      ten ** AEV(i), ten ** WEV(i))) + Stress
        !'A-86
                End If
                RVC(i) = Changevolume(RV(i), YieldSt(i), &
                    Stress + ss_new, Ccp(i), Csp(i))
                Filled(i) = .True.
              Else
                !' Pores are continually empty (WEV < ss_new) ...those pores are dry and still dry after the wetting process
                Filled(i) = .False.
                If ((Assumption .EQ. 1) .And. (Ccp(i) > zero)) Then
                  tmp = ten ** AEV(i)
                  If (Stress > ten ** AEV(i)) Then
                    tmp = CVStress() * ten ** (((Log10(Stress &
                        / ten ** AEV(i)) * ((RefSr ** pm) &
                        * (Ccp(i) - Ccdp(i)) + Ccdp(i))) &
                        / Ccp(i)) + AEV(i))
                  End If
                  If (tmp > YieldSt(i)) Then
                    YieldSt(i) = tmp
                    RVC(i) = Changevolume(RV(i), YieldSt(i), &
                        YieldSt(i), Ccp(i), Csp(i))
                  End If
                End If
              End If
            Else
              !' For all pores that has WEV> current suction (..suction..) - AEV, WEV value are the same...pore increase in volume due to decrease in (suction+stress)
              !' Yield stresses are the same
              If (Stress + ss_new > YieldSt(i)) Then
                YieldSt(i) = Stress + ss_new
              End If
              RVC(i) = Changevolume(RV(i), YieldSt(i), &
                  Stress + ss_new, Ccp(i), Csp(i))
            End If
          END DO
          Suction = ss_new     ! Change current suction to the new soil suction.
        End Subroutine Wetting

        SUBROUTINE Loading(st_new)  !'Increase net mean stress at a certain soil suction
          USE types
          USE parameters
          IMPLICIT NONE
          REAL(DP), intent(in) :: st_new
          INTEGER :: i
          REAL(DP) :: s, sa, ys, ysd, tmp  !' s= real suction

          DO i = 1, Npoint
            s = ten ** WEV(i)        !' set variable s = water entry value of the pore on reference WPD
            If (st_new + s > YieldSt(i)) Then
              ys = st_new + s
            Else
              ys = YieldSt(i)
            End If
            If (Filled(i)) Then   ! For pores that are currently filled with water
              If (st_new + Suction > YieldSt(i)) Then
                YieldSt(i) = st_new + Suction
              End If
              RVC(i) = Changevolume(RV(i), YieldSt(i), &
                  Suction + st_new, Ccp(i), Csp(i))
            Else  !' If pores are not filled with water...
              If ((s / ChangeWetSuction(ys, &
                  ten ** AEV(i), s)) > Suction) Then !' We have to check if WEV of the pore < current suction...then will be filled with water
                YieldSt(i) = ys                                            ! Group of pores must be wetted before apply a load
                RVC(i) = Changevolume(RV(i), YieldSt(i), &
                    Suction + st_new, Ccp(i), Csp(i))
                Filled(i) = .True.
              Else
                !' For all pores that has WEV> current suction (..suction..) - AEV, WEV value are the same...pore increase in volume due to decrease in (suction+stress)
                !' Yield stresses are the same
                If ((Assumption .EQ. 1) .And. (Ccp(i) > 0)) Then
                  tmp = ten ** AEV(i)
                  If (Stress > ten ** AEV(i)) Then
                    tmp = CVStress() * ten ** (((Log10(st_new &
                        / ten ** AEV(i)) * ((RefSr ** pm) &
                        * (Ccp(i) - Ccdp(i)) + Ccdp(i))) &
                        / Ccp(i)) + AEV(i))
                    !'If (Stress < tmp) Then
                    !'  MsgBox "Stress = " + Str(Stress) + "    Yield stress = " + Str(tmp)
                    !'End If
                  End If
                  If (tmp > YieldSt(i)) Then
                    YieldSt(i) = tmp
                    RVC(i) = Changevolume(RV(i), YieldSt(i), &
                        YieldSt(i), Ccp(i), Csp(i))
        !' A-87
                  End If
                End If
                !'Filled(i) = False
                !'AEVC(i) = Log10(ten ** AEV(i) / ChangeSuction(YieldSt(i), ten ** AEV(i)))
                !'WEVC(i) = Log10(ten ** WEV(i) / ChangeSuction(YieldSt(i), ten ** AEV(i)))
              End If
            End If
          END DO
          Stress = st_new !'Change current suction to the new stress.
        End Subroutine Loading

        SUBROUTINE Unloading(st_new)    !Decrease net mean stress at a certain soil suction
          USE types
          USE parameters
          IMPLICIT NONE
          REAL(DP), intent(in) :: st_new
          INTEGER :: i
          REAL(DP) :: s, sa, ys, ysd       !'s= real suction

          DO i = 1, Npoint
            s = ten ** WEV(i)
            ys = YieldSt(i)
            If (Filled(i))Then   !For the pore that is filled with water
              RVC(i) = Changevolume(RV(i), YieldSt(i), &
                  Suction + st_new, Ccp(i), Csp(i))
            End If
          END DO
          Stress = st_new     !'Change current suction to the new stress.
        End SUBROUTINE Unloading


        SUBROUTINE DegreeofsaturationSWCC() !' This procedure is used to calculate Srdry and Srwet
          USE types
          USE parameters
          IMPLICIT NONE
          INTEGER :: i, k
          REAL(DP) :: tmp1, tmp2, intv, cs, ct, X, temp

          CALL SlurryPoreSize       !' Reset the soil to initial slurry condition
          !frmflash.Show
          ct = Stress          !' cs = current stress
          cs = Suction         !' ct = current suction
          intv = (MaxSuction - MinSuction) / NumSr  !' Take equal interval in log scale
          !frmflash.lbprogress.Caption = "Please wait, calculating Ref Drying S% SWCC..."
          !frmflash.Refresh
          DO i = 1, NumSr !' along the drying process
            cs = ten ** (intv * (i - 1) + MinSuction)
            CALL Drying(cs )
            tmp1 = zero
            tmp2 = zero
            DO k = 1, Npoint
              tmp1 = tmp1 + RVC(k)
              If (Filled(k)) Then
                tmp2 = tmp2 + RVC(k)
              End If
            END DO
            Srdry(i) = tmp2 / tmp1
          END DO
          !frmflash.lbprogress.Caption = "Please wait, calculating Ref Wetting S% SWCC..."
          !frmflash.Refresh
          DO i = 1, NumSr !' along the wetting process
            cs = ten ** (MaxSuction - intv * (i - 1))
            CALL Wetting(cs)
            tmp1 = zero
            tmp2 = zero
            DO k = 1,Npoint
              tmp1 = tmp1 + RVC(k)
              If (Filled(k)) Then
                tmp2 = tmp2 + (RVC(k) / Gs)
              End If
            END DO
            Srwet(NumSr - i + 1) = tmp2 * Gs / tmp1
          END DO
          !frmflash.Refresh
        End SUBROUTINE DegreeofsaturationSWCC

        Function RfSr_value(curvetype, ssvalue)
          USE types
          USE parameters
          IMPLICIT NONE
          INTEGER :: i, curvetype
          REAL(DP) :: ssvalue, RfSr_value

          !'curvetype=1: drying; = 2: scanning; =3: wetting

          If (curvetype .EQ. 1) Then
            RfSr_value = Srdry(int((Log10(ssvalue) - MinSuction) &
                / ((MaxSuction - MinSuction) / NumSr)) + 1)
          End If
          If (curvetype .EQ. 3) Then
            RfSr_value = Srwet(int((Log10(ssvalue) - MinSuction) &
                / ((MaxSuction - MinSuction) / NumSr)) + 1)
          End If
        !'A-88

        End Function RfSr_value


        SUBROUTINE Calresults()
          USE types
          USE parameters
          IMPLICIT NONE
          INTEGER :: i, j, k
          REAL(DP) :: tmp1, tmp2, intv, cs, ct, X, temp
          CALL initialize_datapoint()
          CALL initialize_SrDry_SrWet()
          CALL initialize_f()
          CALL initialize_stp()

          errocc = .False.
          If (Assumption .EQ. 1) Then
            CALL DegreeofsaturationSWCC
          Else
            CONTINUE
            !frmflash.Show
          End If
          CALL SlurryPoreSize
          ct = Stress          !' = current stress
          cs = Suction         !' = current suction
          stpn = 0
          datapointn = 0

          DO i = 1, nsteps
            If (.NOT. ist(i)) Then
              stpn = stpn + 1
              intv = (Log10(vl(i)) - Log10(cs)) / (npp(i) - one) !' Take equal interval in log scale
              startpoint(stpn) = datapointn + 1
              DO j = 1, npp(i)
                datapointn = datapointn + 1
                !frmflash.lbprogress.Caption = "Calculating data point #" + Str(datapointn)
                !frmflash.Refresh
                ss(datapointn) = ten ** ((j - 1) * intv &
                    + Log10(cs))
                st(datapointn) = ct
                If (intv > 0) Then
                  CALL Drying(ss(datapointn))
                Else
                  CALL Wetting(ss(datapointn))
                End If
                tmp1 = zero
                tmp2 = zero
                DO k = 1, Npoint
                  tmp1 = tmp1 + RVC(k)
                  If (Filled(k)) Then
                    If (Airentrapped(k)) Then
                       tmp2 = tmp2 + (RVC(k) / Gs) * (one - beta)
                     Else
                      tmp2 = tmp2 + (RVC(k) / Gs)
                    End If
                  End If
                END DO
                ee(datapointn) = tmp1
                w(datapointn) = tmp2
              END DO
              cs = vl(i)
              endpoint(stpn) = datapointn
            End If
            If (ist(i)) Then
              stpn = stpn + 1
              intv = (Log10(vl(i)) - Log10(ct)) / (npp(i) - one)       !' Take equal interval in log scale
              startpoint(stpn) = datapointn + 1
              DO j = 1, npp(i)
                datapointn = datapointn + 1
                !frmflash.lbprogress.Caption = "         Calculating data point #" + Str(datapointn)
                !frmflash.Refresh
                ss(datapointn) = cs
                st(datapointn) = ten ** ((j - 1) * intv &
                  + Log10(ct))
                If (intv > zero) Then
                  CALL Loading(st(datapointn) * CVStress())
                Else
                  CALL Unloading(st(datapointn) * CVStress())
                End If
                tmp1 = zero
                tmp2 = zero
                DO k = 1, Npoint
                  tmp1 = tmp1 + RVC(k)
        !'A-89
                  If (Filled(k)) Then
                    If (Airentrapped(k)) Then
                      tmp2 = tmp2 + (RVC(k) / Gs) * (one - beta)
                    Else
                      tmp2 = tmp2 + (RVC(k) / Gs)
                    End If
                  End If
                END DO
                ee(datapointn) = tmp1
                w(datapointn) = tmp2
              END DO
              ct = vl(i)
              endpoint(stpn) = datapointn
            End If
          END DO
          DO i = 1, datapointn
            If (ss(i) >= 999999.0_DP) Then
              ss(i) = 999998.0_DP
            END IF
            If (st(i) >= 999999.0_DP) Then
              st(i) = 999998.0_DP
            END IF
            Sr(i) = w(i) * Gs / ee(i)
            vw(i) = Sr(i) * ee(i) &
                / (ee(i) + 1)
          END DO
          !Unload frmflash
          If (errocc) Then
            !MsgBox " Input data is not valid, please check the PORE-SHAPE PARAMETER"
            !print *, " Input data is not valid, please check &
            !    the PORE-SHAPE PARAMETER"
          End If
        End SUBROUTINE Calresults


        SUBROUTINE save_to_file()
        ! saves ss, st, ee, w, Sr, vw to a file called "epus_results.csv"
          IMPLICIT NONE
          INTEGER, parameter :: out_unit=20
          INTEGER :: i, j, k

          open (unit=out_unit,file="epus_results.csv",action="write",status="replace")
          write(out_unit, *) "st,ss,e,w,Sr,vw"

          DO i = 1, npts
          !"(6(ES13.4))"
          !fmt='(*(G0.4,:,","))')
          !fmt="(*(g0:','))")
            write (out_unit,fmt='(*(G0.4,:,","))') ss(i), &
                                               st(i), &
                                               ee(i), &
                                               w(i), &
                                               Sr(i), &
                                               vw(i)
          END DO
          close (out_unit)

        END SUBROUTINE



        SUBROUTINE dealloc()
        !Deallocate epus arrays.  This is useful if using epus more than
        ! once in the same program run.  run dealloc between the each use
        ! otherwise you will get an 'array already allocated error'.
          IF (ALLOCATED (ss)) DEALLOCATE (ss)
          IF (ALLOCATED (st)) DEALLOCATE (st)
          IF (ALLOCATED (ee)) DEALLOCATE (ee)
          IF (ALLOCATED (w)) DEALLOCATE (w)
          IF (ALLOCATED (Sr)) DEALLOCATE (Sr)
          IF (ALLOCATED (vw)) DEALLOCATE (vw)

          IF (ALLOCATED (ist)) DEALLOCATE (ist)
          IF (ALLOCATED (vl)) DEALLOCATE (vl)
          IF (ALLOCATED (npp)) DEALLOCATE (npp)

          IF (ALLOCATED (startpoint)) DEALLOCATE (startpoint)
          IF (ALLOCATED (endpoint)) DEALLOCATE (endpoint)

          IF (ALLOCATED (SrDry)) DEALLOCATE (SrDry)
          IF (ALLOCATED (SrWet)) DEALLOCATE (SrWet)

          IF (ALLOCATED (AEV)) DEALLOCATE (AEV)
          IF (ALLOCATED (WEV)) DEALLOCATE (WEV)
          IF (ALLOCATED (Ccp)) DEALLOCATE (Ccp)
          IF (ALLOCATED (Csp)) DEALLOCATE (Csp)
          IF (ALLOCATED (Ccdp)) DEALLOCATE (Ccdp)
          IF (ALLOCATED (RV)) DEALLOCATE (RV)
          IF (ALLOCATED (YieldSt)) DEALLOCATE (YieldSt)
          IF (ALLOCATED (Filled)) DEALLOCATE (Filled)
          IF (ALLOCATED (RVC)) DEALLOCATE (RVC)
          IF (ALLOCATED (AEVC)) DEALLOCATE (AEVC)
          IF (ALLOCATED (WEVC)) DEALLOCATE (WEVC)
          IF (ALLOCATED (Airentrapped)) DEALLOCATE (Airentrapped)


        END SUBROUTINE





        SUBROUTINE print_module_var()
        ! prints epus module data to stdout (usually the screen)
          IMPLICIT NONE

          PRINT *, "*******Module variables******"
          PRINT *, "stpn ", stpn
          PRINT *, "ist ", ist
          PRINT *, "vl ", vl
          PRINT *, "npp ", npp
          PRINT *, "Npoint ", Npoint
          PRINT *, "npts ", npts

          Print *, "wsat ", wsat
          Print *, "a ", a
          Print *, "b ", b
          Print *, "wr ", wr
          Print *, "sl ", sl
          Print *, "logDS ", logDS
          Print *, "logRS ", logRS
          Print *, "Ccs ", Ccs
          Print *, "Css ", Css
          Print *, "Ccd ", Ccd
          Print *, "Gs ", Gs
          Print *, "Assumption ", Assumption
          Print *, "beta ", beta
          Print *, "pm ", pm
          Print *, "Pore_shape ", Pore_shape
          Print *, "K0 ", K0
          Print *, "soilname ", soilname
          Print *, "username ", username
          Print *, "Npoint ", Npoint
          Print *, "NumSr ", NumSr
          Print *, "MaxSuction ", MaxSuction
          Print *, "MinSuction ", MinSuction
          PRINT *, "*************************"
        END SUBROUTINE print_module_var
      END MODULE




      PROGRAM main
      !This is an example of how to use the epus module
      ! steps for use are basically:
      ! 1. Assign all variables (see code for which variables are needed)
      ! 2. Allocate and fill the stress path arrays ist, vl, npp
      ! 3. Run subroutine Calresults to calculate.
      ! 4. Results for ss, st, w, ee, Sr, vw are now avalable.  You can
      !    use the startpoint and endpoint arrays to break up the
      !    results into individual stress path steps.
      ! 5. if needed run subroutine dealloc. to deallocate all the arrays.

        USE types
        USE parameters
        USE epus
        IMPLICIT NONE

        NumSr=400
        Npoint=1000
        nsteps=5
        ALLOCATE(ist(1:nsteps))
        ALLOCATE(vl(1:nsteps))
        ALLOCATE(npp(1:nsteps))
        ist(1:5) = (/.true., .true., .false., .false., .false./)
        npp(1:5) = 100
        vl(1:5) = (/ 20._DP, 1._DP, 1.E6_DP, 30._DP, 1500._DP/)


        wsat=0.262_DP
        a=3.1E6_DP
        b=3.377_DP
        wr=0.128_DP
        sl=0.115_DP
        logDS=0.6_DP
        logRS=2._DP
        Ccs=0.115_DP
        Css=0.019_DP
        Ccd=0._DP
        Gs=2.7_DP
        Assumption=1
        beta=0.1_DP
        pm=1._DP
        Pore_shape=1._DP
        K0=1._DP
        soilname="unknown"
        username="unknown"
        Npoint=1000!400
        NumSr=400
        MaxSuction=6._DP
        MinSuction=-3._DP

        CALL print_module_var()
        CALL Calresults()
        CALL save_to_file()


      END PROGRAM