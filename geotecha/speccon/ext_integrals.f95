      MODULE types
        IMPLICIT NONE      
        INTEGER, PARAMETER :: DP = KIND(0.0D0)
      END MODULE



      SUBROUTINE dim1sin_af_linear(m, at, ab, zt, zb, a, neig, nlayers)
        USE types
        IMPLICIT NONE

        INTEGER, intent(in) :: neig
        INTEGER, intent(in) :: nlayers
        REAL(DP), intent(in), dimension(0:neig-1) ::m
        REAL(DP), intent(in), dimension(0:nlayers-1) :: at
        REAL(DP), intent(in), dimension(0:nlayers-1) :: ab
        REAL(DP), intent(in), dimension(0:nlayers-1) :: zt
        REAL(DP), intent(in), dimension(0:nlayers-1) :: zb
        REAL(DP), intent(out), dimension(0:neig-1, 0:neig-1) :: a
        INTEGER :: i , j, layer
        REAL(DP) :: a_slope

        a=0.0D0
        DO layer = 0, nlayers-1
          a_slope = (ab(layer) - at(layer)) / (zb(layer) - zt(layer))
          DO j = 0, neig-1
              i=j
      a(i, i) = a(i, i) + ((1.0d0/4.0d0)*a_slope*m(i)**(-2)*sin(m(i)*zb(&
      layer))**2 - 1.0d0/4.0d0*a_slope*m(i)**(-2)*sin(m(i)*zt(layer))**&
      2 + (1.0d0/2.0d0)*a_slope*1.0/m(i)*cos(m(i)*zb(layer))*sin(m(i)*&
      zb(layer))*zt(layer) - 1.0d0/2.0d0*a_slope*1.0/m(i)*zb(layer)*cos&
      (m(i)*zb(layer))*sin(m(i)*zb(layer)) - 1.0d0/2.0d0*a_slope*zb(&
      layer)*sin(m(i)*zb(layer))**2*zt(layer) - 1.0d0/2.0d0*a_slope*zb(&
      layer)*cos(m(i)*zb(layer))**2*zt(layer) + (1.0d0/4.0d0)*a_slope*&
      zb(layer)**2*sin(m(i)*zb(layer))**2 + (1.0d0/4.0d0)*a_slope*zb(&
      layer)**2*cos(m(i)*zb(layer))**2 + (1.0d0/4.0d0)*a_slope*zt(layer&
      )**2*sin(m(i)*zt(layer))**2 + (1.0d0/4.0d0)*a_slope*zt(layer)**2*&
      cos(m(i)*zt(layer))**2 - 1.0d0/2.0d0*1.0/m(i)*cos(m(i)*zb(layer))&
      *sin(m(i)*zb(layer))*at(layer) + (1.0d0/2.0d0)*1.0/m(i)*cos(m(i)*&
      zt(layer))*sin(m(i)*zt(layer))*at(layer) + (1.0d0/2.0d0)*zb(layer&
      )*sin(m(i)*zb(layer))**2*at(layer) + (1.0d0/2.0d0)*zb(layer)*cos(&
      m(i)*zb(layer))**2*at(layer) - 1.0d0/2.0d0*zt(layer)*sin(m(i)*zt(&
      layer))**2*at(layer) - 1.0d0/2.0d0*zt(layer)*cos(m(i)*zt(layer))&
      **2*at(layer))
            DO i = j+1, neig-1
      a(i, j) = a(i, j) + (a_slope*m(i)**2*1.0/(m(i)**4 - 2*m(j)**2*m(i)&
      **2 + m(j)**4)*sin(m(i)*zb(layer))*sin(m(j)*zb(layer)) - a_slope*&
      m(i)**2*1.0/(m(i)**4 - 2*m(j)**2*m(i)**2 + m(j)**4)*sin(m(i)*zt(&
      layer))*sin(m(j)*zt(layer)) + a_slope*m(i)**3*1.0/(m(i)**4 - 2*m(&
      j)**2*m(i)**2 + m(j)**4)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer))*&
      zt(layer) - a_slope*m(i)**3*1.0/(m(i)**4 - 2*m(j)**2*m(i)**2 + m(&
      j)**4)*zb(layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) + 2*&
      a_slope*m(j)*m(i)*1.0/(m(i)**4 - 2*m(j)**2*m(i)**2 + m(j)**4)*cos&
      (m(i)*zb(layer))*cos(m(j)*zb(layer)) - 2*a_slope*m(j)*m(i)*1.0/(m&
      (i)**4 - 2*m(j)**2*m(i)**2 + m(j)**4)*cos(m(i)*zt(layer))*cos(m(j&
      )*zt(layer)) - a_slope*m(j)*m(i)**2*1.0/(m(i)**4 - 2*m(j)**2*m(i)&
      **2 + m(j)**4)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer))*zt(layer)&
      + a_slope*m(j)*m(i)**2*1.0/(m(i)**4 - 2*m(j)**2*m(i)**2 + m(j)**4&
      )*zb(layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) + a_slope*m(j&
      )**2*1.0/(m(i)**4 - 2*m(j)**2*m(i)**2 + m(j)**4)*sin(m(i)*zb(&
      layer))*sin(m(j)*zb(layer)) - a_slope*m(j)**2*1.0/(m(i)**4 - 2*m(&
      j)**2*m(i)**2 + m(j)**4)*sin(m(i)*zt(layer))*sin(m(j)*zt(layer))&
      - a_slope*m(j)**2*m(i)*1.0/(m(i)**4 - 2*m(j)**2*m(i)**2 + m(j)**4&
      )*cos(m(i)*zb(layer))*sin(m(j)*zb(layer))*zt(layer) + a_slope*m(j&
      )**2*m(i)*1.0/(m(i)**4 - 2*m(j)**2*m(i)**2 + m(j)**4)*zb(layer)*&
      cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) + a_slope*m(j)**3*1.0/(m(&
      i)**4 - 2*m(j)**2*m(i)**2 + m(j)**4)*cos(m(j)*zb(layer))*sin(m(i)&
      *zb(layer))*zt(layer) - a_slope*m(j)**3*1.0/(m(i)**4 - 2*m(j)**2*&
      m(i)**2 + m(j)**4)*zb(layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(&
      layer)) - m(i)**3*1.0/(m(i)**4 - 2*m(j)**2*m(i)**2 + m(j)**4)*cos&
      (m(i)*zb(layer))*sin(m(j)*zb(layer))*at(layer) + m(i)**3*1.0/(m(i&
      )**4 - 2*m(j)**2*m(i)**2 + m(j)**4)*cos(m(i)*zt(layer))*sin(m(j)*&
      zt(layer))*at(layer) + m(j)*m(i)**2*1.0/(m(i)**4 - 2*m(j)**2*m(i)&
      **2 + m(j)**4)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer))*at(layer)&
      - m(j)*m(i)**2*1.0/(m(i)**4 - 2*m(j)**2*m(i)**2 + m(j)**4)*cos(m(&
      j)*zt(layer))*sin(m(i)*zt(layer))*at(layer) + m(j)**2*m(i)*1.0/(m&
      (i)**4 - 2*m(j)**2*m(i)**2 + m(j)**4)*cos(m(i)*zb(layer))*sin(m(j&
      )*zb(layer))*at(layer) - m(j)**2*m(i)*1.0/(m(i)**4 - 2*m(j)**2*m(&
      i)**2 + m(j)**4)*cos(m(i)*zt(layer))*sin(m(j)*zt(layer))*at(layer&
      ) - m(j)**3*1.0/(m(i)**4 - 2*m(j)**2*m(i)**2 + m(j)**4)*cos(m(j)*&
      zb(layer))*sin(m(i)*zb(layer))*at(layer) + m(j)**3*1.0/(m(i)**4 -&
      2*m(j)**2*m(i)**2 + m(j)**4)*cos(m(j)*zt(layer))*sin(m(i)*zt(&
      layer))*at(layer))
            END DO
          END DO
        END DO

        DO j = 0, neig -2
          DO i = j + 1, neig-1
            a(j,i) = a(i, j)
          END DO
        END DO

      END SUBROUTINE



      SUBROUTINE dim1sin_abf_linear(m, at, ab, bt, bb, zt, zb, a, &
                                    neig, nlayers)
        USE types
        IMPLICIT NONE

        INTEGER, intent(in) :: neig
        INTEGER, intent(in) :: nlayers
        REAL(DP), intent(in), dimension(0:neig-1) ::m
        REAL(DP), intent(in), dimension(0:nlayers-1) :: at,ab,bt,bb,zt,zb
!        REAL(DP), intent(in), dimension(0:nlayers-1) :: at
!        REAL(DP), intent(in), dimension(0:nlayers-1) :: ab
!        REAL(DP), intent(in), dimension(0:nlayers-1) :: bt
!        REAL(DP), intent(in), dimension(0:nlayers-1) :: bb
!        REAL(DP), intent(in), dimension(0:nlayers-1) :: zt
!        REAL(DP), intent(in), dimension(0:nlayers-1) :: zb
        REAL(DP), intent(out), dimension(0:neig-1, 0:neig-1) :: a
        INTEGER :: i , j, layer
        REAL(DP) :: a_slope, b_slope


        a=0.0D0
        DO layer = 0, nlayers-1
          a_slope = (ab(layer) - at(layer)) / (zb(layer) - zt(layer))
          b_slope = (bb(layer) - bt(layer)) / (zb(layer) - zt(layer))
          DO j = 0, neig-1
              i=j
      a(i, i) = a(i, i) + ((1.0d0/4.0d0)*a_slope*b_slope*m(i)**(-3)*cos(&
      m(i)*zb(layer))*sin(m(i)*zb(layer)) - 1.0d0/4.0d0*a_slope*b_slope&
      *m(i)**(-3)*cos(m(i)*zt(layer))*sin(m(i)*zt(layer)) - 1.0d0/2.0d0&
      *a_slope*b_slope*m(i)**(-2)*sin(m(i)*zb(layer))**2*zt(layer) + (&
      1.0d0/4.0d0)*a_slope*b_slope*m(i)**(-2)*zb(layer)*sin(m(i)*zb(&
      layer))**2 - 1.0d0/4.0d0*a_slope*b_slope*m(i)**(-2)*zb(layer)*cos&
      (m(i)*zb(layer))**2 + (1.0d0/4.0d0)*a_slope*b_slope*m(i)**(-2)*zt&
      (layer)*sin(m(i)*zt(layer))**2 + (1.0d0/4.0d0)*a_slope*b_slope*m(&
      i)**(-2)*zt(layer)*cos(m(i)*zt(layer))**2 - 1.0d0/2.0d0*a_slope*&
      b_slope*1.0/m(i)*cos(m(i)*zb(layer))*sin(m(i)*zb(layer))*zt(layer&
      )**2 + a_slope*b_slope*1.0/m(i)*zb(layer)*cos(m(i)*zb(layer))*sin&
      (m(i)*zb(layer))*zt(layer) - 1.0d0/2.0d0*a_slope*b_slope*1.0/m(i)&
      *zb(layer)**2*cos(m(i)*zb(layer))*sin(m(i)*zb(layer)) + (1.0d0/&
      2.0d0)*a_slope*b_slope*zb(layer)*sin(m(i)*zb(layer))**2*zt(layer)&
      **2 + (1.0d0/2.0d0)*a_slope*b_slope*zb(layer)*cos(m(i)*zb(layer))&
      **2*zt(layer)**2 - 1.0d0/2.0d0*a_slope*b_slope*zb(layer)**2*sin(m&
      (i)*zb(layer))**2*zt(layer) - 1.0d0/2.0d0*a_slope*b_slope*zb(&
      layer)**2*cos(m(i)*zb(layer))**2*zt(layer) + (1.0d0/6.0d0)*&
      a_slope*b_slope*zb(layer)**3*sin(m(i)*zb(layer))**2 + (1.0d0/&
      6.0d0)*a_slope*b_slope*zb(layer)**3*cos(m(i)*zb(layer))**2 -&
      1.0d0/6.0d0*a_slope*b_slope*zt(layer)**3*sin(m(i)*zt(layer))**2 -&
      1.0d0/6.0d0*a_slope*b_slope*zt(layer)**3*cos(m(i)*zt(layer))**2 +&
      (1.0d0/4.0d0)*a_slope*m(i)**(-2)*sin(m(i)*zb(layer))**2*bt(layer&
      ) - 1.0d0/4.0d0*a_slope*m(i)**(-2)*sin(m(i)*zt(layer))**2*bt(&
      layer) + (1.0d0/2.0d0)*a_slope*1.0/m(i)*cos(m(i)*zb(layer))*sin(m&
      (i)*zb(layer))*zt(layer)*bt(layer) - 1.0d0/2.0d0*a_slope*1.0/m(i)&
      *zb(layer)*cos(m(i)*zb(layer))*sin(m(i)*zb(layer))*bt(layer) -&
      1.0d0/2.0d0*a_slope*zb(layer)*sin(m(i)*zb(layer))**2*zt(layer)*bt&
      (layer) - 1.0d0/2.0d0*a_slope*zb(layer)*cos(m(i)*zb(layer))**2*zt&
      (layer)*bt(layer) + (1.0d0/4.0d0)*a_slope*zb(layer)**2*sin(m(i)*&
      zb(layer))**2*bt(layer) + (1.0d0/4.0d0)*a_slope*zb(layer)**2*cos(&
      m(i)*zb(layer))**2*bt(layer) + (1.0d0/4.0d0)*a_slope*zt(layer)**2&
      *sin(m(i)*zt(layer))**2*bt(layer) + (1.0d0/4.0d0)*a_slope*zt(&
      layer)**2*cos(m(i)*zt(layer))**2*bt(layer) + (1.0d0/4.0d0)*&
      b_slope*m(i)**(-2)*sin(m(i)*zb(layer))**2*at(layer) - 1.0d0/4.0d0&
      *b_slope*m(i)**(-2)*sin(m(i)*zt(layer))**2*at(layer) + (1.0d0/&
      2.0d0)*b_slope*1.0/m(i)*cos(m(i)*zb(layer))*sin(m(i)*zb(layer))*&
      zt(layer)*at(layer) - 1.0d0/2.0d0*b_slope*1.0/m(i)*zb(layer)*cos(&
      m(i)*zb(layer))*sin(m(i)*zb(layer))*at(layer) - 1.0d0/2.0d0*&
      b_slope*zb(layer)*sin(m(i)*zb(layer))**2*zt(layer)*at(layer) -&
      1.0d0/2.0d0*b_slope*zb(layer)*cos(m(i)*zb(layer))**2*zt(layer)*at&
      (layer) + (1.0d0/4.0d0)*b_slope*zb(layer)**2*sin(m(i)*zb(layer))&
      **2*at(layer) + (1.0d0/4.0d0)*b_slope*zb(layer)**2*cos(m(i)*zb(&
      layer))**2*at(layer) + (1.0d0/4.0d0)*b_slope*zt(layer)**2*sin(m(i&
      )*zt(layer))**2*at(layer) + (1.0d0/4.0d0)*b_slope*zt(layer)**2*&
      cos(m(i)*zt(layer))**2*at(layer) - 1.0d0/2.0d0*1.0/m(i)*cos(m(i)*&
      zb(layer))*sin(m(i)*zb(layer))*bt(layer)*at(layer) + (1.0d0/2.0d0&
      )*1.0/m(i)*cos(m(i)*zt(layer))*sin(m(i)*zt(layer))*bt(layer)*at(&
      layer) + (1.0d0/2.0d0)*zb(layer)*sin(m(i)*zb(layer))**2*bt(layer)&
      *at(layer) + (1.0d0/2.0d0)*zb(layer)*cos(m(i)*zb(layer))**2*bt(&
      layer)*at(layer) - 1.0d0/2.0d0*zt(layer)*sin(m(i)*zt(layer))**2*&
      bt(layer)*at(layer) - 1.0d0/2.0d0*zt(layer)*cos(m(i)*zt(layer))**&
      2*bt(layer)*at(layer))
            DO i = j+1, neig-1
      a(i, j) = a(i, j) + (2*a_slope*b_slope*m(i)**3*1.0/(m(i)**6 - 3*m(&
      j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(m(i)*zb(layer))*&
      sin(m(j)*zb(layer)) - 2*a_slope*b_slope*m(i)**3*1.0/(m(i)**6 - 3*&
      m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(m(i)*zt(layer&
      ))*sin(m(j)*zt(layer)) - 2*a_slope*b_slope*m(i)**4*1.0/(m(i)**6 -&
      3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*sin(m(i)*zb(&
      layer))*sin(m(j)*zb(layer))*zt(layer) + 2*a_slope*b_slope*m(i)**4&
      *1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*&
      zb(layer)*sin(m(i)*zb(layer))*sin(m(j)*zb(layer)) - a_slope*&
      b_slope*m(i)**5*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)&
      **2 - m(j)**6)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer))*zt(layer)&
      **2 + 2*a_slope*b_slope*m(i)**5*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4&
      + 3*m(j)**4*m(i)**2 - m(j)**6)*zb(layer)*cos(m(i)*zb(layer))*sin(&
      m(j)*zb(layer))*zt(layer) - a_slope*b_slope*m(i)**5*1.0/(m(i)**6&
      - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*zb(layer)**2*&
      cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) - 6*a_slope*b_slope*m(j)*&
      m(i)**2*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(&
      j)**6)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) + 6*a_slope*&
      b_slope*m(j)*m(i)**2*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4&
      *m(i)**2 - m(j)**6)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer)) - 4*&
      a_slope*b_slope*m(j)*m(i)**3*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3&
      *m(j)**4*m(i)**2 - m(j)**6)*cos(m(i)*zb(layer))*cos(m(j)*zb(layer&
      ))*zt(layer) + 4*a_slope*b_slope*m(j)*m(i)**3*1.0/(m(i)**6 - 3*m(&
      j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*zb(layer)*cos(m(i)*&
      zb(layer))*cos(m(j)*zb(layer)) + a_slope*b_slope*m(j)*m(i)**4*1.0&
      /(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(&
      m(j)*zb(layer))*sin(m(i)*zb(layer))*zt(layer)**2 - 2*a_slope*&
      b_slope*m(j)*m(i)**4*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4&
      *m(i)**2 - m(j)**6)*zb(layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(&
      layer))*zt(layer) + a_slope*b_slope*m(j)*m(i)**4*1.0/(m(i)**6 - 3&
      *m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*zb(layer)**2*cos(&
      m(j)*zb(layer))*sin(m(i)*zb(layer)) + 6*a_slope*b_slope*m(j)**2*m&
      (i)*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**&
      6)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) - 6*a_slope*b_slope*m(&
      j)**2*m(i)*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 -&
      m(j)**6)*cos(m(i)*zt(layer))*sin(m(j)*zt(layer)) + 2*a_slope*&
      b_slope*m(j)**2*m(i)**3*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)&
      **4*m(i)**2 - m(j)**6)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer))*zt&
      (layer)**2 - 4*a_slope*b_slope*m(j)**2*m(i)**3*1.0/(m(i)**6 - 3*m&
      (j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*zb(layer)*cos(m(i)*&
      zb(layer))*sin(m(j)*zb(layer))*zt(layer) + 2*a_slope*b_slope*m(j)&
      **2*m(i)**3*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2&
      - m(j)**6)*zb(layer)**2*cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) -&
      2*a_slope*b_slope*m(j)**3*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(&
      j)**4*m(i)**2 - m(j)**6)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer))&
      + 2*a_slope*b_slope*m(j)**3*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*&
      m(j)**4*m(i)**2 - m(j)**6)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer&
      )) + 4*a_slope*b_slope*m(j)**3*m(i)*1.0/(m(i)**6 - 3*m(j)**2*m(i)&
      **4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(m(i)*zb(layer))*cos(m(j)*&
      zb(layer))*zt(layer) - 4*a_slope*b_slope*m(j)**3*m(i)*1.0/(m(i)**&
      6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*zb(layer)*&
      cos(m(i)*zb(layer))*cos(m(j)*zb(layer)) - 2*a_slope*b_slope*m(j)&
      **3*m(i)**2*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2&
      - m(j)**6)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer))*zt(layer)**2 +&
      4*a_slope*b_slope*m(j)**3*m(i)**2*1.0/(m(i)**6 - 3*m(j)**2*m(i)**&
      4 + 3*m(j)**4*m(i)**2 - m(j)**6)*zb(layer)*cos(m(j)*zb(layer))*&
      sin(m(i)*zb(layer))*zt(layer) - 2*a_slope*b_slope*m(j)**3*m(i)**2&
      *1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*&
      zb(layer)**2*cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) + 2*a_slope*&
      b_slope*m(j)**4*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)&
      **2 - m(j)**6)*sin(m(i)*zb(layer))*sin(m(j)*zb(layer))*zt(layer)&
      - 2*a_slope*b_slope*m(j)**4*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*&
      m(j)**4*m(i)**2 - m(j)**6)*zb(layer)*sin(m(i)*zb(layer))*sin(m(j)&
      *zb(layer)) - a_slope*b_slope*m(j)**4*m(i)*1.0/(m(i)**6 - 3*m(j)&
      **2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(m(i)*zb(layer))*&
      sin(m(j)*zb(layer))*zt(layer)**2 + 2*a_slope*b_slope*m(j)**4*m(i)&
      *1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*&
      zb(layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer))*zt(layer) -&
      a_slope*b_slope*m(j)**4*m(i)*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3&
      *m(j)**4*m(i)**2 - m(j)**6)*zb(layer)**2*cos(m(i)*zb(layer))*sin(&
      m(j)*zb(layer)) + a_slope*b_slope*m(j)**5*1.0/(m(i)**6 - 3*m(j)**&
      2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(m(j)*zb(layer))*sin(&
      m(i)*zb(layer))*zt(layer)**2 - 2*a_slope*b_slope*m(j)**5*1.0/(m(i&
      )**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*zb(layer)&
      *cos(m(j)*zb(layer))*sin(m(i)*zb(layer))*zt(layer) + a_slope*&
      b_slope*m(j)**5*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)&
      **2 - m(j)**6)*zb(layer)**2*cos(m(j)*zb(layer))*sin(m(i)*zb(layer&
      )) + a_slope*m(i)**4*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4&
      *m(i)**2 - m(j)**6)*sin(m(i)*zb(layer))*sin(m(j)*zb(layer))*bt(&
      layer) - a_slope*m(i)**4*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j&
      )**4*m(i)**2 - m(j)**6)*sin(m(i)*zt(layer))*sin(m(j)*zt(layer))*&
      bt(layer) + a_slope*m(i)**5*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*&
      m(j)**4*m(i)**2 - m(j)**6)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer&
      ))*zt(layer)*bt(layer) - a_slope*m(i)**5*1.0/(m(i)**6 - 3*m(j)**2&
      *m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*zb(layer)*cos(m(i)*zb(&
      layer))*sin(m(j)*zb(layer))*bt(layer) + 2*a_slope*m(j)*m(i)**3*&
      1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*&
      cos(m(i)*zb(layer))*cos(m(j)*zb(layer))*bt(layer) - 2*a_slope*m(j&
      )*m(i)**3*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 -&
      m(j)**6)*cos(m(i)*zt(layer))*cos(m(j)*zt(layer))*bt(layer) -&
      a_slope*m(j)*m(i)**4*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4&
      *m(i)**2 - m(j)**6)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer))*zt(&
      layer)*bt(layer) + a_slope*m(j)*m(i)**4*1.0/(m(i)**6 - 3*m(j)**2*&
      m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*zb(layer)*cos(m(j)*zb(&
      layer))*sin(m(i)*zb(layer))*bt(layer) - 2*a_slope*m(j)**2*m(i)**3&
      *1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*&
      cos(m(i)*zb(layer))*sin(m(j)*zb(layer))*zt(layer)*bt(layer) + 2*&
      a_slope*m(j)**2*m(i)**3*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)&
      **4*m(i)**2 - m(j)**6)*zb(layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(&
      layer))*bt(layer) - 2*a_slope*m(j)**3*m(i)*1.0/(m(i)**6 - 3*m(j)&
      **2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(m(i)*zb(layer))*&
      cos(m(j)*zb(layer))*bt(layer) + 2*a_slope*m(j)**3*m(i)*1.0/(m(i)&
      **6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(m(i)*&
      zt(layer))*cos(m(j)*zt(layer))*bt(layer) + 2*a_slope*m(j)**3*m(i)&
      **2*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**&
      6)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer))*zt(layer)*bt(layer) -&
      2*a_slope*m(j)**3*m(i)**2*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(&
      j)**4*m(i)**2 - m(j)**6)*zb(layer)*cos(m(j)*zb(layer))*sin(m(i)*&
      zb(layer))*bt(layer) - a_slope*m(j)**4*1.0/(m(i)**6 - 3*m(j)**2*m&
      (i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*sin(m(i)*zb(layer))*sin(m(j&
      )*zb(layer))*bt(layer) + a_slope*m(j)**4*1.0/(m(i)**6 - 3*m(j)**2&
      *m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*sin(m(i)*zt(layer))*sin(m&
      (j)*zt(layer))*bt(layer) + a_slope*m(j)**4*m(i)*1.0/(m(i)**6 - 3*&
      m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(m(i)*zb(layer&
      ))*sin(m(j)*zb(layer))*zt(layer)*bt(layer) - a_slope*m(j)**4*m(i)&
      *1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*&
      zb(layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer))*bt(layer) -&
      a_slope*m(j)**5*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)&
      **2 - m(j)**6)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer))*zt(layer)*&
      bt(layer) + a_slope*m(j)**5*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*&
      m(j)**4*m(i)**2 - m(j)**6)*zb(layer)*cos(m(j)*zb(layer))*sin(m(i)&
      *zb(layer))*bt(layer) + b_slope*m(i)**4*1.0/(m(i)**6 - 3*m(j)**2*&
      m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*sin(m(i)*zb(layer))*sin(m(&
      j)*zb(layer))*at(layer) - b_slope*m(i)**4*1.0/(m(i)**6 - 3*m(j)**&
      2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*sin(m(i)*zt(layer))*sin(&
      m(j)*zt(layer))*at(layer) + b_slope*m(i)**5*1.0/(m(i)**6 - 3*m(j)&
      **2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(m(i)*zb(layer))*&
      sin(m(j)*zb(layer))*zt(layer)*at(layer) - b_slope*m(i)**5*1.0/(m(&
      i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*zb(layer&
      )*cos(m(i)*zb(layer))*sin(m(j)*zb(layer))*at(layer) + 2*b_slope*m&
      (j)*m(i)**3*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2&
      - m(j)**6)*cos(m(i)*zb(layer))*cos(m(j)*zb(layer))*at(layer) - 2*&
      b_slope*m(j)*m(i)**3*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4&
      *m(i)**2 - m(j)**6)*cos(m(i)*zt(layer))*cos(m(j)*zt(layer))*at(&
      layer) - b_slope*m(j)*m(i)**4*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 +&
      3*m(j)**4*m(i)**2 - m(j)**6)*cos(m(j)*zb(layer))*sin(m(i)*zb(&
      layer))*zt(layer)*at(layer) + b_slope*m(j)*m(i)**4*1.0/(m(i)**6 -&
      3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*zb(layer)*cos(m(&
      j)*zb(layer))*sin(m(i)*zb(layer))*at(layer) - 2*b_slope*m(j)**2*m&
      (i)**3*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j&
      )**6)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer))*zt(layer)*at(layer&
      ) + 2*b_slope*m(j)**2*m(i)**3*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 +&
      3*m(j)**4*m(i)**2 - m(j)**6)*zb(layer)*cos(m(i)*zb(layer))*sin(m(&
      j)*zb(layer))*at(layer) - 2*b_slope*m(j)**3*m(i)*1.0/(m(i)**6 - 3&
      *m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(m(i)*zb(layer&
      ))*cos(m(j)*zb(layer))*at(layer) + 2*b_slope*m(j)**3*m(i)*1.0/(m(&
      i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(m(i)&
      *zt(layer))*cos(m(j)*zt(layer))*at(layer) + 2*b_slope*m(j)**3*m(i&
      )**2*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)&
      **6)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer))*zt(layer)*at(layer)&
      - 2*b_slope*m(j)**3*m(i)**2*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*&
      m(j)**4*m(i)**2 - m(j)**6)*zb(layer)*cos(m(j)*zb(layer))*sin(m(i)&
      *zb(layer))*at(layer) - b_slope*m(j)**4*1.0/(m(i)**6 - 3*m(j)**2*&
      m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*sin(m(i)*zb(layer))*sin(m(&
      j)*zb(layer))*at(layer) + b_slope*m(j)**4*1.0/(m(i)**6 - 3*m(j)**&
      2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*sin(m(i)*zt(layer))*sin(&
      m(j)*zt(layer))*at(layer) + b_slope*m(j)**4*m(i)*1.0/(m(i)**6 - 3&
      *m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(m(i)*zb(layer&
      ))*sin(m(j)*zb(layer))*zt(layer)*at(layer) - b_slope*m(j)**4*m(i)&
      *1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*&
      zb(layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer))*at(layer) -&
      b_slope*m(j)**5*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)&
      **2 - m(j)**6)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer))*zt(layer)*&
      at(layer) + b_slope*m(j)**5*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*&
      m(j)**4*m(i)**2 - m(j)**6)*zb(layer)*cos(m(j)*zb(layer))*sin(m(i)&
      *zb(layer))*at(layer) - m(i)**5*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4&
      + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(m(i)*zb(layer))*sin(m(j)*zb(&
      layer))*bt(layer)*at(layer) + m(i)**5*1.0/(m(i)**6 - 3*m(j)**2*m(&
      i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(m(i)*zt(layer))*sin(m(j)&
      *zt(layer))*bt(layer)*at(layer) + m(j)*m(i)**4*1.0/(m(i)**6 - 3*m&
      (j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(m(j)*zb(layer))&
      *sin(m(i)*zb(layer))*bt(layer)*at(layer) - m(j)*m(i)**4*1.0/(m(i)&
      **6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(m(j)*&
      zt(layer))*sin(m(i)*zt(layer))*bt(layer)*at(layer) + 2*m(j)**2*m(&
      i)**3*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)&
      **6)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer))*bt(layer)*at(layer)&
      - 2*m(j)**2*m(i)**3*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*&
      m(i)**2 - m(j)**6)*cos(m(i)*zt(layer))*sin(m(j)*zt(layer))*bt(&
      layer)*at(layer) - 2*m(j)**3*m(i)**2*1.0/(m(i)**6 - 3*m(j)**2*m(i&
      )**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(m(j)*zb(layer))*sin(m(i)*&
      zb(layer))*bt(layer)*at(layer) + 2*m(j)**3*m(i)**2*1.0/(m(i)**6 -&
      3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(m(j)*zt(&
      layer))*sin(m(i)*zt(layer))*bt(layer)*at(layer) - m(j)**4*m(i)*&
      1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*&
      cos(m(i)*zb(layer))*sin(m(j)*zb(layer))*bt(layer)*at(layer) + m(j&
      )**4*m(i)*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 -&
      m(j)**6)*cos(m(i)*zt(layer))*sin(m(j)*zt(layer))*bt(layer)*at(&
      layer) + m(j)**5*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i&
      )**2 - m(j)**6)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer))*bt(layer)&
      *at(layer) - m(j)**5*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4&
      *m(i)**2 - m(j)**6)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer))*bt(&
      layer)*at(layer))
            END DO
          END DO
        END DO

        DO j = 0, neig -2
          DO i = j + 1, neig-1
            a(j,i) = a(i, j)
          END DO
        END DO

      END SUBROUTINE



      SUBROUTINE dim1sin_D_aDf_linear(m, at, ab, zt, zb, a, neig, nlayers)
        USE types
        IMPLICIT NONE

        INTEGER, intent(in) :: neig
        INTEGER, intent(in) :: nlayers
        REAL(DP), intent(in), dimension(0:neig-1) ::m
        REAL(DP), intent(in), dimension(0:nlayers-1) :: at
        REAL(DP), intent(in), dimension(0:nlayers-1) :: ab
        REAL(DP), intent(in), dimension(0:nlayers-1) :: zt
        REAL(DP), intent(in), dimension(0:nlayers-1) :: zb
        REAL(DP), intent(out), dimension(0:neig-1, 0:neig-1) :: a
        INTEGER :: i , j, layer
        REAL(DP) :: a_slope

        a=0.0D0
        DO layer = 0, nlayers-1
          a_slope = (ab(layer) - at(layer)) / (zb(layer) - zt(layer))
          DO j = 0, neig-1
              i=j
      a(i, i) = a(i, i) + (m(i)**2*(-1.0d0/4.0d0*a_slope*m(i)**(-2)*sin(&
      m(i)*zt(layer))**2 - 1.0d0/4.0d0*a_slope*zt(layer)**2*sin(m(i)*zt&
      (layer))**2 - 1.0d0/4.0d0*a_slope*zt(layer)**2*cos(m(i)*zt(layer&
      ))**2 + (1.0d0/2.0d0)*1.0/m(i)*cos(m(i)*zt(layer))*sin(m(i)*zt(&
      layer))*at(layer) + (1.0d0/2.0d0)*zt(layer)*sin(m(i)*zt(layer))**&
      2*at(layer) + (1.0d0/2.0d0)*zt(layer)*cos(m(i)*zt(layer))**2*at(&
      layer)) - m(i)**2*(-1.0d0/4.0d0*a_slope*m(i)**(-2)*sin(m(i)*zb(&
      layer))**2 - 1.0d0/2.0d0*a_slope*1.0/m(i)*cos(m(i)*zb(layer))*sin&
      (m(i)*zb(layer))*zt(layer) + (1.0d0/2.0d0)*a_slope*1.0/m(i)*zb(&
      layer)*cos(m(i)*zb(layer))*sin(m(i)*zb(layer)) - 1.0d0/2.0d0*&
      a_slope*zb(layer)*sin(m(i)*zb(layer))**2*zt(layer) - 1.0d0/2.0d0*&
      a_slope*zb(layer)*cos(m(i)*zb(layer))**2*zt(layer) + (1.0d0/4.0d0&
      )*a_slope*zb(layer)**2*sin(m(i)*zb(layer))**2 + (1.0d0/4.0d0)*&
      a_slope*zb(layer)**2*cos(m(i)*zb(layer))**2 + (1.0d0/2.0d0)*1.0/m&
      (i)*cos(m(i)*zb(layer))*sin(m(i)*zb(layer))*at(layer) + (1.0d0/&
      2.0d0)*zb(layer)*sin(m(i)*zb(layer))**2*at(layer) + (1.0d0/2.0d0)&
      *zb(layer)*cos(m(i)*zb(layer))**2*at(layer)))
            DO i = j+1, neig-1
      a(i, j) = a(i, j) + (m(j)*m(i)*(a_slope*m(i)**2*1.0/(m(i)**4 - 2*m&
      (j)**2*m(i)**2 + m(j)**4)*cos(m(i)*zt(layer))*cos(m(j)*zt(layer&
      )) + 2*a_slope*m(j)*m(i)*1.0/(m(i)**4 - 2*m(j)**2*m(i)**2 + m(j)&
      **4)*sin(m(i)*zt(layer))*sin(m(j)*zt(layer)) + a_slope*m(j)**2*&
      1.0/(m(i)**4 - 2*m(j)**2*m(i)**2 + m(j)**4)*cos(m(i)*zt(layer))*&
      cos(m(j)*zt(layer)) + m(i)**3*1.0/(m(i)**4 - 2*m(j)**2*m(i)**2 +&
      m(j)**4)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer))*at(layer) - m(j)&
      *m(i)**2*1.0/(m(i)**4 - 2*m(j)**2*m(i)**2 + m(j)**4)*cos(m(i)*zt(&
      layer))*sin(m(j)*zt(layer))*at(layer) - m(j)**2*m(i)*1.0/(m(i)**4&
      - 2*m(j)**2*m(i)**2 + m(j)**4)*cos(m(j)*zt(layer))*sin(m(i)*zt(&
      layer))*at(layer) + m(j)**3*1.0/(m(i)**4 - 2*m(j)**2*m(i)**2 + m(&
      j)**4)*cos(m(i)*zt(layer))*sin(m(j)*zt(layer))*at(layer)) - m(j)*&
      m(i)*(a_slope*m(i)**2*1.0/(m(i)**4 - 2*m(j)**2*m(i)**2 + m(j)**4)&
      *cos(m(i)*zb(layer))*cos(m(j)*zb(layer)) - a_slope*m(i)**3*1.0/(m&
      (i)**4 - 2*m(j)**2*m(i)**2 + m(j)**4)*cos(m(j)*zb(layer))*sin(m(i&
      )*zb(layer))*zt(layer) + a_slope*m(i)**3*1.0/(m(i)**4 - 2*m(j)**2&
      *m(i)**2 + m(j)**4)*zb(layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(&
      layer)) + 2*a_slope*m(j)*m(i)*1.0/(m(i)**4 - 2*m(j)**2*m(i)**2 +&
      m(j)**4)*sin(m(i)*zb(layer))*sin(m(j)*zb(layer)) + a_slope*m(j)*m&
      (i)**2*1.0/(m(i)**4 - 2*m(j)**2*m(i)**2 + m(j)**4)*cos(m(i)*zb(&
      layer))*sin(m(j)*zb(layer))*zt(layer) - a_slope*m(j)*m(i)**2*1.0/&
      (m(i)**4 - 2*m(j)**2*m(i)**2 + m(j)**4)*zb(layer)*cos(m(i)*zb(&
      layer))*sin(m(j)*zb(layer)) + a_slope*m(j)**2*1.0/(m(i)**4 - 2*m(&
      j)**2*m(i)**2 + m(j)**4)*cos(m(i)*zb(layer))*cos(m(j)*zb(layer))&
      + a_slope*m(j)**2*m(i)*1.0/(m(i)**4 - 2*m(j)**2*m(i)**2 + m(j)**4&
      )*cos(m(j)*zb(layer))*sin(m(i)*zb(layer))*zt(layer) - a_slope*m(j&
      )**2*m(i)*1.0/(m(i)**4 - 2*m(j)**2*m(i)**2 + m(j)**4)*zb(layer)*&
      cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) - a_slope*m(j)**3*1.0/(m(&
      i)**4 - 2*m(j)**2*m(i)**2 + m(j)**4)*cos(m(i)*zb(layer))*sin(m(j)&
      *zb(layer))*zt(layer) + a_slope*m(j)**3*1.0/(m(i)**4 - 2*m(j)**2*&
      m(i)**2 + m(j)**4)*zb(layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(&
      layer)) + m(i)**3*1.0/(m(i)**4 - 2*m(j)**2*m(i)**2 + m(j)**4)*cos&
      (m(j)*zb(layer))*sin(m(i)*zb(layer))*at(layer) - m(j)*m(i)**2*1.0&
      /(m(i)**4 - 2*m(j)**2*m(i)**2 + m(j)**4)*cos(m(i)*zb(layer))*sin(&
      m(j)*zb(layer))*at(layer) - m(j)**2*m(i)*1.0/(m(i)**4 - 2*m(j)**2&
      *m(i)**2 + m(j)**4)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer))*at(&
      layer) + m(j)**3*1.0/(m(i)**4 - 2*m(j)**2*m(i)**2 + m(j)**4)*cos(&
      m(i)*zb(layer))*sin(m(j)*zb(layer))*at(layer)))
            END DO
          END DO
        END DO

        DO j = 0, neig -2
          DO i = j + 1, neig-1
            a(j,i) = a(i, j)
          END DO
        END DO

      END SUBROUTINE


      SUBROUTINE dim1sin_ab_linear(m, at, ab, bt, bb, zt, zb, &
                                   a, neig, nlayers)
        USE types
        IMPLICIT NONE

        INTEGER, intent(in) :: neig
        INTEGER, intent(in) :: nlayers
        REAL(DP), intent(in), dimension(0:neig-1) ::m
        REAL(DP), intent(in), dimension(0:nlayers-1) :: at,ab,bt,bb,zt,zb
        REAL(DP), intent(out), dimension(0:neig-1) :: a
        INTEGER :: i, layer
        REAL(DP) :: a_slope, b_slope
        a=0.0D0
        DO layer = 0, nlayers-1
          a_slope = (ab(layer) - at(layer)) / (zb(layer) - zt(layer))
          b_slope = (bb(layer) - bt(layer)) / (zb(layer) - zt(layer))
          DO i = 0, neig-1
      a(i) = a(i) + (2*a_slope*b_slope*m(i)**(-3)*cos(m(i)*zb(layer)) -&
      2*a_slope*b_slope*m(i)**(-3)*cos(m(i)*zt(layer)) - 2*a_slope*&
      b_slope*m(i)**(-2)*sin(m(i)*zb(layer))*zt(layer) + 2*a_slope*&
      b_slope*m(i)**(-2)*zb(layer)*sin(m(i)*zb(layer)) - a_slope*&
      b_slope*1.0/m(i)*cos(m(i)*zb(layer))*zt(layer)**2 + 2*a_slope*&
      b_slope*1.0/m(i)*zb(layer)*cos(m(i)*zb(layer))*zt(layer) -&
      a_slope*b_slope*1.0/m(i)*zb(layer)**2*cos(m(i)*zb(layer)) +&
      a_slope*m(i)**(-2)*sin(m(i)*zb(layer))*bt(layer) - a_slope*m(i)**&
      (-2)*sin(m(i)*zt(layer))*bt(layer) + a_slope*1.0/m(i)*cos(m(i)*zb&
      (layer))*zt(layer)*bt(layer) - a_slope*1.0/m(i)*zb(layer)*cos(m(i&
      )*zb(layer))*bt(layer) + b_slope*m(i)**(-2)*sin(m(i)*zb(layer))*&
      at(layer) - b_slope*m(i)**(-2)*sin(m(i)*zt(layer))*at(layer) +&
      b_slope*1.0/m(i)*cos(m(i)*zb(layer))*zt(layer)*at(layer) -&
      b_slope*1.0/m(i)*zb(layer)*cos(m(i)*zb(layer))*at(layer) - 1.0/m(&
      i)*cos(m(i)*zb(layer))*bt(layer)*at(layer) + 1.0/m(i)*cos(m(i)*zt&
      (layer))*bt(layer)*at(layer))
          END DO
        END DO

      END SUBROUTINE


      SUBROUTINE dim1sin_abc_linear(m, at, ab, bt, bb, ct, cb, &
                                    zt, zb, a, neig, nlayers)
        USE types
        IMPLICIT NONE

        INTEGER, intent(in) :: neig
        INTEGER, intent(in) :: nlayers
        REAL(DP), intent(in), dimension(0:neig-1) ::m
        REAL(DP), intent(in), dimension(0:nlayers-1) :: at,ab,bt,bb, &
                                                        ct,cb,zt,zb
        REAL(DP), intent(out), dimension(0:neig-1) :: a
        INTEGER :: i, layer
        REAL(DP) :: a_slope, b_slope, c_slope
        a=0.0D0
        DO layer = 0, nlayers-1
          a_slope = (ab(layer) - at(layer)) / (zb(layer) - zt(layer))
          b_slope = (bb(layer) - bt(layer)) / (zb(layer) - zt(layer))
          c_slope = (cb(layer) - ct(layer)) / (zb(layer) - zt(layer))
          DO i = 0, neig-1
      a(i) = a(i) + (-6*a_slope*b_slope*c_slope*m(i)**(-4)*sin(m(i)*zb(&
      layer)) + 6*a_slope*b_slope*c_slope*m(i)**(-4)*sin(m(i)*zt(layer&
      )) - 6*a_slope*b_slope*c_slope*m(i)**(-3)*cos(m(i)*zb(layer))*zt(&
      layer) + 6*a_slope*b_slope*c_slope*m(i)**(-3)*zb(layer)*cos(m(i)*&
      zb(layer)) + 3*a_slope*b_slope*c_slope*m(i)**(-2)*sin(m(i)*zb(&
      layer))*zt(layer)**2 - 6*a_slope*b_slope*c_slope*m(i)**(-2)*zb(&
      layer)*sin(m(i)*zb(layer))*zt(layer) + 3*a_slope*b_slope*c_slope*&
      m(i)**(-2)*zb(layer)**2*sin(m(i)*zb(layer)) + a_slope*b_slope*&
      c_slope*1.0/m(i)*cos(m(i)*zb(layer))*zt(layer)**3 - 3*a_slope*&
      b_slope*c_slope*1.0/m(i)*zb(layer)*cos(m(i)*zb(layer))*zt(layer)&
      **2 + 3*a_slope*b_slope*c_slope*1.0/m(i)*zb(layer)**2*cos(m(i)*zb&
      (layer))*zt(layer) - a_slope*b_slope*c_slope*1.0/m(i)*zb(layer)**&
      3*cos(m(i)*zb(layer)) + 2*a_slope*b_slope*m(i)**(-3)*cos(m(i)*zb(&
      layer))*ct(layer) - 2*a_slope*b_slope*m(i)**(-3)*cos(m(i)*zt(&
      layer))*ct(layer) - 2*a_slope*b_slope*m(i)**(-2)*sin(m(i)*zb(&
      layer))*zt(layer)*ct(layer) + 2*a_slope*b_slope*m(i)**(-2)*zb(&
      layer)*sin(m(i)*zb(layer))*ct(layer) - a_slope*b_slope*1.0/m(i)*&
      cos(m(i)*zb(layer))*zt(layer)**2*ct(layer) + 2*a_slope*b_slope*&
      1.0/m(i)*zb(layer)*cos(m(i)*zb(layer))*zt(layer)*ct(layer) -&
      a_slope*b_slope*1.0/m(i)*zb(layer)**2*cos(m(i)*zb(layer))*ct(&
      layer) + 2*a_slope*c_slope*m(i)**(-3)*cos(m(i)*zb(layer))*bt(&
      layer) - 2*a_slope*c_slope*m(i)**(-3)*cos(m(i)*zt(layer))*bt(&
      layer) - 2*a_slope*c_slope*m(i)**(-2)*sin(m(i)*zb(layer))*zt(&
      layer)*bt(layer) + 2*a_slope*c_slope*m(i)**(-2)*zb(layer)*sin(m(i&
      )*zb(layer))*bt(layer) - a_slope*c_slope*1.0/m(i)*cos(m(i)*zb(&
      layer))*zt(layer)**2*bt(layer) + 2*a_slope*c_slope*1.0/m(i)*zb(&
      layer)*cos(m(i)*zb(layer))*zt(layer)*bt(layer) - a_slope*c_slope*&
      1.0/m(i)*zb(layer)**2*cos(m(i)*zb(layer))*bt(layer) + a_slope*m(i&
      )**(-2)*sin(m(i)*zb(layer))*ct(layer)*bt(layer) - a_slope*m(i)**(&
      -2)*sin(m(i)*zt(layer))*ct(layer)*bt(layer) + a_slope*1.0/m(i)*&
      cos(m(i)*zb(layer))*zt(layer)*ct(layer)*bt(layer) - a_slope*1.0/m&
      (i)*zb(layer)*cos(m(i)*zb(layer))*ct(layer)*bt(layer) + 2*b_slope&
      *c_slope*m(i)**(-3)*cos(m(i)*zb(layer))*at(layer) - 2*b_slope*&
      c_slope*m(i)**(-3)*cos(m(i)*zt(layer))*at(layer) - 2*b_slope*&
      c_slope*m(i)**(-2)*sin(m(i)*zb(layer))*zt(layer)*at(layer) + 2*&
      b_slope*c_slope*m(i)**(-2)*zb(layer)*sin(m(i)*zb(layer))*at(layer&
      ) - b_slope*c_slope*1.0/m(i)*cos(m(i)*zb(layer))*zt(layer)**2*at(&
      layer) + 2*b_slope*c_slope*1.0/m(i)*zb(layer)*cos(m(i)*zb(layer))&
      *zt(layer)*at(layer) - b_slope*c_slope*1.0/m(i)*zb(layer)**2*cos(&
      m(i)*zb(layer))*at(layer) + b_slope*m(i)**(-2)*sin(m(i)*zb(layer&
      ))*ct(layer)*at(layer) - b_slope*m(i)**(-2)*sin(m(i)*zt(layer))*&
      ct(layer)*at(layer) + b_slope*1.0/m(i)*cos(m(i)*zb(layer))*zt(&
      layer)*ct(layer)*at(layer) - b_slope*1.0/m(i)*zb(layer)*cos(m(i)*&
      zb(layer))*ct(layer)*at(layer) + c_slope*m(i)**(-2)*sin(m(i)*zb(&
      layer))*bt(layer)*at(layer) - c_slope*m(i)**(-2)*sin(m(i)*zt(&
      layer))*bt(layer)*at(layer) + c_slope*1.0/m(i)*cos(m(i)*zb(layer&
      ))*zt(layer)*bt(layer)*at(layer) - c_slope*1.0/m(i)*zb(layer)*cos&
      (m(i)*zb(layer))*bt(layer)*at(layer) - 1.0/m(i)*cos(m(i)*zb(layer&
      ))*ct(layer)*bt(layer)*at(layer) + 1.0/m(i)*cos(m(i)*zt(layer))*&
      ct(layer)*bt(layer)*at(layer))
          END DO
        END DO

      END SUBROUTINE
     
     
      SUBROUTINE dim1sin_D_aDb_linear(m, at, ab, bt, bb, zt, zb, &
                                      a, neig, nlayers)
        USE types
        IMPLICIT NONE

        INTEGER, intent(in) :: neig
        INTEGER, intent(in) :: nlayers
        REAL(DP), intent(in), dimension(0:neig-1) ::m
        REAL(DP), intent(in), dimension(0:nlayers-1) :: at,ab,bt,bb,zt,zb
        REAL(DP), intent(out), dimension(0:neig-1) :: a
        INTEGER :: i, layer
        REAL(DP) :: a_slope, b_slope
        a=0.0D0
        DO layer = 0, nlayers-1
          a_slope = (ab(layer) - at(layer)) / (zb(layer) - zt(layer))
          b_slope = (bb(layer) - bt(layer)) / (zb(layer) - zt(layer))
          DO i = 0, neig-1
      a(i) = a(i) + (b_slope*m(i)*(a_slope*m(i)**(-2)*cos(zt(layer)*m(i&
      )) + at(layer)*1.0/m(i)*sin(zt(layer)*m(i))) - b_slope*m(i)*(&
      a_slope*m(i)**(-2)*cos(zb(layer)*m(i)) + a_slope*zb(layer)*1.0/m(&
      i)*sin(zb(layer)*m(i)) - a_slope*zt(layer)*1.0/m(i)*sin(zb(layer)&
      *m(i)) + at(layer)*1.0/m(i)*sin(zb(layer)*m(i))))
          END DO
        END DO

        DO i = 0, neig-1
      a(i) = a(i) + (-1.0/(zb(0) - zt(0))*(bb(0) - bt(0))*at(0)*sin(zt(0&
      )*m(i)) + 1.0/(zb(nlayers - 1) - zt(nlayers - 1))*(bb(nlayers - 1&
      ) - bt(nlayers - 1))*ab(nlayers - 1)*sin(zb(nlayers - 1)*m(i)))
        END DO

      END SUBROUTINE



      SUBROUTINE eload_linear(loadtim, loadmag, eigs, tvals,&
                              dT, a, neig, nload, nt)
        USE types
        IMPLICIT NONE

        INTEGER, intent(in) :: neig
        INTEGER, intent(in) :: nload
        INTEGER, intent(in) :: nt
        REAL(DP), intent(in), dimension(0:nload-1) :: loadtim
        REAL(DP), intent(in), dimension(0:nload-1) :: loadmag
        REAL(DP), intent(in), dimension(0:neig-1) :: eigs
        REAL(DP), intent(in), dimension(0:nt-1) :: tvals
        REAL(DP), intent(in) :: dT
        REAL(DP), intent(out), dimension(0:nt-1, 0:neig-1) :: a
        INTEGER :: i , j, k
        REAL(DP):: EPSILON
        a=0.0D0
        EPSILON = 0.0000005D0
        DO i = 0, nt-1
          DO k = 0, nload-2

            IF (tvals(i) < loadtim(k)) EXIT !t is before load step

            IF (tvals(i) >= loadtim(k + 1)) THEN
              !t is after the load step
              IF(ABS(loadtim(k) - loadtim(k + 1)) <= &
                (ABS(loadtim(k) + loadtim(k + 1))*EPSILON)) THEN
                !step load
                CONTINUE
              ELSEIF(ABS(loadmag(k) - loadmag(k + 1)) <= &
                    (ABS(loadmag(k) + loadmag(k + 1))*EPSILON)) THEN
                !constant load
                DO j=0, neig-1
      a(i, j) = a(i, j) + (1.0/eigs(j)*exp(-dT*eigs(j)*(-loadtim(k + 1)&
      + tvals(i)))*loadmag(k)/dT - 1.0/eigs(j)*exp(-dT*eigs(j)*(&
      -loadtim(k) + tvals(i)))*loadmag(k)/dT)
                END DO
              ELSE
                !ramp load
                DO j=0, neig-1
      a(i, j) = a(i, j) + (-1.0/eigs(j)*(dT*eigs(j)*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k))*(-loadtim(k) + tvals(i))*&
      exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k + 1) - dT*&
      eigs(j)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k))*&
      (-loadtim(k) + tvals(i))*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i&
      )))*loadmag(k) - dT*eigs(j)*loadtim(k + 1)*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k))*exp(-dT*eigs(j)*(-loadtim&
      (k) + tvals(i)))*loadmag(k) + dT*eigs(j)*loadtim(k)*1.0/(-dT*eigs&
      (j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k))*exp(-dT*eigs(j)*(&
      -loadtim(k) + tvals(i)))*loadmag(k + 1) - dT*tvals(i)*eigs(j)*1.0&
      /(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k))*exp(-dT*&
      eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k + 1) + dT*tvals(i)*&
      eigs(j)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k))*&
      exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k) + 1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k))*exp(-dT*eigs(j)*(&
      -loadtim(k) + tvals(i)))*loadmag(k + 1) - 1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k))*exp(-dT*eigs(j)*(-loadtim&
      (k) + tvals(i)))*loadmag(k))/dT + 1.0/eigs(j)*(dT*eigs(j)*(&
      -loadtim(k + 1) + tvals(i))*exp(-dT*eigs(j)*(-loadtim(k + 1) +&
      tvals(i)))*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k&
      ))*loadmag(k + 1) - dT*eigs(j)*(-loadtim(k + 1) + tvals(i))*exp(&
      -dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k))*loadmag(k) + dT*eigs(j)*&
      exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*loadtim(k)*1.0/(-dT&
      *eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k))*loadmag(k + 1) -&
      dT*eigs(j)*loadtim(k + 1)*exp(-dT*eigs(j)*(-loadtim(k + 1) +&
      tvals(i)))*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k&
      ))*loadmag(k) - dT*tvals(i)*eigs(j)*exp(-dT*eigs(j)*(-loadtim(k +&
      1) + tvals(i)))*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k))*loadmag(k + 1) + dT*tvals(i)*eigs(j)*exp(-dT*eigs(j)*&
      (-loadtim(k + 1) + tvals(i)))*1.0/(-dT*eigs(j)*loadtim(k + 1) +&
      dT*eigs(j)*loadtim(k))*loadmag(k) + exp(-dT*eigs(j)*(-loadtim(k +&
      1) + tvals(i)))*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k))*loadmag(k + 1) - exp(-dT*eigs(j)*(-loadtim(k + 1) +&
      tvals(i)))*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k&
      ))*loadmag(k))/dT)
                END DO
              END IF
            ELSE
              !t is in the load step
              IF(ABS(loadmag(k) - loadmag(k + 1)) <= &
                    (ABS(loadmag(k) + loadmag(k + 1))*EPSILON)) THEN
                !constant load
                DO j=0, neig-1
      a(i, j) = a(i, j) + (-1.0/eigs(j)*exp(-dT*eigs(j)*(-loadtim(k) +&
      tvals(i)))*loadmag(k)/dT + 1.0/eigs(j)*loadmag(k)/dT)
                END DO
              ELSE
                !ramp load
                DO j=0, neig-1
      a(i, j) = a(i, j) + (1.0/eigs(j)*(-dT*eigs(j)*loadtim(k + 1)*1.0/(&
      -dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k))*loadmag(k) +&
      dT*eigs(j)*loadtim(k)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j&
      )*loadtim(k))*loadmag(k + 1) - dT*tvals(i)*eigs(j)*1.0/(-dT*eigs(&
      j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k))*loadmag(k + 1) + dT*&
      tvals(i)*eigs(j)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k))*loadmag(k) + 1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*&
      eigs(j)*loadtim(k))*loadmag(k + 1) - 1.0/(-dT*eigs(j)*loadtim(k +&
      1) + dT*eigs(j)*loadtim(k))*loadmag(k))/dT - 1.0/eigs(j)*(dT*eigs&
      (j)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k))*(&
      -loadtim(k) + tvals(i))*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))&
      *loadmag(k + 1) - dT*eigs(j)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT&
      *eigs(j)*loadtim(k))*(-loadtim(k) + tvals(i))*exp(-dT*eigs(j)*(&
      -loadtim(k) + tvals(i)))*loadmag(k) - dT*eigs(j)*loadtim(k + 1)*&
      1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k))*exp(-dT*&
      eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k) + dT*eigs(j)*loadtim&
      (k)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k))*exp(&
      -dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k + 1) - dT*tvals(i&
      )*eigs(j)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k&
      ))*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k + 1) + dT*&
      tvals(i)*eigs(j)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k))*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k)&
      + 1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k))*exp(&
      -dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k + 1) - 1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k))*exp(-dT*eigs(j)*(&
      -loadtim(k) + tvals(i)))*loadmag(k))/dT)
                END DO
              END IF
            END IF
          END DO
        END DO

      END SUBROUTINE


      SUBROUTINE edload_linear(loadtim, loadmag, eigs, tvals,&
                               dT, a, neig, nload, nt)
        USE types
        IMPLICIT NONE

        INTEGER, intent(in) :: neig
        INTEGER, intent(in) :: nload
        INTEGER, intent(in) :: nt
        REAL(DP), intent(in), dimension(0:nload-1) :: loadtim
        REAL(DP), intent(in), dimension(0:nload-1) :: loadmag
        REAL(DP), intent(in), dimension(0:neig-1) :: eigs
        REAL(DP), intent(in), dimension(0:nt-1) :: tvals
        REAL(DP), intent(in) :: dT
        REAL(DP), intent(out), dimension(0:nt-1, 0:neig-1) :: a
        INTEGER :: i , j, k
        REAL(DP):: EPSILON
        a=0.0D0
        EPSILON = 0.0000005D0
        DO i = 0, nt-1
          DO k = 0, nload-2

            IF (tvals(i) < loadtim(k)) EXIT !t is before load step

            IF (tvals(i) >= loadtim(k + 1)) THEN
              !t is after the load step
              IF(ABS(loadtim(k) - loadtim(k + 1)) <= &
                (ABS(loadtim(k) + loadtim(k + 1))*EPSILON)) THEN
                !step load
                DO j=0, neig-1
      a(i, j) = a(i, j) + (exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*(&
      loadmag(k + 1) - loadmag(k)))
                END DO
              ELSEIF(ABS(loadmag(k) - loadmag(k + 1)) <= &
                    (ABS(loadmag(k) + loadmag(k + 1))*EPSILON)) THEN
                !constant load
                CONTINUE
              ELSE
                !ramp load
                DO j=0, neig-1
      a(i, j) = a(i, j) + (-1.0/eigs(j)*1.0/(loadtim(k + 1) - loadtim(k&
      ))*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*(loadmag(k + 1) -&
      loadmag(k))/dT + 1.0/eigs(j)*exp(-dT*eigs(j)*(-loadtim(k + 1) +&
      tvals(i)))*1.0/(loadtim(k + 1) - loadtim(k))*(loadmag(k + 1) -&
      loadmag(k))/dT)
                END DO
              END IF
            ELSE
              !t is in the load step
              IF(ABS(loadmag(k) - loadmag(k + 1)) <= &
                    (ABS(loadmag(k) + loadmag(k + 1))*EPSILON)) THEN
                !constant load
                CONTINUE
              ELSE
                !ramp load
                DO j=0, neig-1
      a(i, j) = a(i, j) + (1.0/eigs(j)*1.0/(loadtim(k + 1) - loadtim(k))&
      *(loadmag(k + 1) - loadmag(k))/dT - 1.0/eigs(j)*1.0/(loadtim(k +&
      1) - loadtim(k))*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*(&
      loadmag(k + 1) - loadmag(k))/dT)
                END DO
              END IF
            END IF
          END DO
        END DO

      END SUBROUTINE



    



      SUBROUTINE eload_coslinear(loadtim, loadmag, omega, phase, &
                                 eigs, tvals, dT, a, neig, nload, nt)
        USE types
        IMPLICIT NONE

        INTEGER, intent(in) :: neig
        INTEGER, intent(in) :: nload
        INTEGER, intent(in) :: nt
        REAL(DP), intent(in), dimension(0:nload-1) :: loadtim
        REAL(DP), intent(in), dimension(0:nload-1) :: loadmag
        REAL(DP), intent(in), dimension(0:neig-1) :: eigs
        REAL(DP), intent(in), dimension(0:nt-1) :: tvals
        REAL(DP), intent(in) :: dT
        REAL(DP), intent(in) :: omega
        REAL(DP), intent(in) :: phase
        REAL(DP), intent(out), dimension(0:nt-1, 0:neig-1) :: a
        INTEGER :: i , j, k
        REAL(DP):: EPSILON
        a=0.0D0
        EPSILON = 0.0000005D0
        DO i = 0, nt-1
          DO k = 0, nload-2

            IF (tvals(i) < loadtim(k)) EXIT !t is before load step

            IF (tvals(i) >= loadtim(k + 1)) THEN
              !t is after the load step
              IF(ABS(loadtim(k) - loadtim(k + 1)) <= &
                (ABS(loadtim(k) + loadtim(k + 1))*EPSILON)) THEN
                !step load
                CONTINUE
              ELSEIF(ABS(loadmag(k) - loadmag(k + 1)) <= &
                    (ABS(loadmag(k) + loadmag(k + 1))*EPSILON)) THEN
                !constant load
                DO j=0, neig-1
      a(i, j) = a(i, j) + (1.0/eigs(j)*(1.0/(1 + omega**2*eigs(j)**(-2)/&
      dT**2)*cos(-omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(i) +&
      phase)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i))) + omega*1.0/&
      eigs(j)*1.0/(1 + omega**2*eigs(j)**(-2)/dT**2)*exp(-dT*eigs(j)*(&
      -loadtim(k + 1) + tvals(i)))*sin(-omega*(-loadtim(k + 1) + tvals(&
      i)) + omega*tvals(i) + phase)/dT)*loadmag(k)/dT - 1.0/eigs(j)*(&
      1.0/(1 + omega**2*eigs(j)**(-2)/dT**2)*cos(-omega*(-loadtim(k) +&
      tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k)&
      + tvals(i))) + omega*1.0/eigs(j)*1.0/(1 + omega**2*eigs(j)**(-2)/&
      dT**2)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(&
      -loadtim(k) + tvals(i)) + omega*tvals(i) + phase)/dT)*loadmag(k)/&
      dT)
                END DO
              ELSE
                !ramp load
                DO j=0, neig-1
      a(i, j) = a(i, j) + (1.0/eigs(j)*(dT*eigs(j)*(-loadtim(k + 1) +&
      tvals(i))*cos(-omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(i&
      ) + phase)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*1.0/(-dT&
      *eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*loadmag(k + 1) - dT*eigs(j)*(-loadtim(k&
      + 1) + tvals(i))*cos(-omega*(-loadtim(k + 1) + tvals(i)) + omega*&
      tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*&
      1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega&
      **2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k) + dT*eigs(j)*&
      cos(-omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase)&
      *exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*loadtim(k)*1.0/(&
      -dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*&
      1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)&
      /dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs&
      (j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1) - dT*eigs(j)*loadtim(k&
      + 1)*cos(-omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(i) +&
      phase)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*loadmag(k) - dT*tvals(i)*eigs(j)*cos(&
      -omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase)*exp&
      (-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*loadmag(k + 1) + dT*tvals(i)*eigs(j)*cos(-omega&
      *(-loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*&
      eigs(j)*(-loadtim(k + 1) + tvals(i)))*1.0/(-dT*eigs(j)*loadtim(k&
      + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k +&
      1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**&
      (-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT&
      **3)*loadmag(k) + omega*(-loadtim(k + 1) + tvals(i))*exp(-dT*eigs&
      (j)*(-loadtim(k + 1) + tvals(i)))*sin(-omega*(-loadtim(k + 1) +&
      tvals(i)) + omega*tvals(i) + phase)*1.0/(-dT*eigs(j)*loadtim(k +&
      1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1&
      )/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(&
      -3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**&
      3)*loadmag(k + 1) - omega*(-loadtim(k + 1) + tvals(i))*exp(-dT*&
      eigs(j)*(-loadtim(k + 1) + tvals(i)))*sin(-omega*(-loadtim(k + 1&
      ) + tvals(i)) + omega*tvals(i) + phase)*1.0/(-dT*eigs(j)*loadtim(&
      k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k&
      + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)&
      **(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/&
      dT**3)*loadmag(k) + omega*exp(-dT*eigs(j)*(-loadtim(k + 1) +&
      tvals(i)))*sin(-omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(&
      i) + phase)*loadtim(k)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(&
      j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*&
      omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      loadmag(k + 1) - omega*loadtim(k + 1)*exp(-dT*eigs(j)*(-loadtim(k&
      + 1) + tvals(i)))*sin(-omega*(-loadtim(k + 1) + tvals(i)) + omega&
      *tvals(i) + phase)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k) -&
      omega*tvals(i)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*sin(&
      -omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase)*1.0&
      /(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2&
      *1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k&
      )/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*&
      eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1) + omega*tvals(i)*&
      exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*sin(-omega*(&
      -loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase)*1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*loadmag(k) + cos(-omega*(-loadtim(k + 1&
      ) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim&
      (k + 1) + tvals(i)))*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)&
      *loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega&
      **2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k&
      + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k +&
      1) - cos(-omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(i) +&
      phase)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*loadmag(k) + omega**2*1.0/eigs(j)*(&
      -loadtim(k + 1) + tvals(i))*cos(-omega*(-loadtim(k + 1) + tvals(i&
      )) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k + 1) +&
      tvals(i)))*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k&
      ) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/&
      eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT&
      **3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1)/dT&
      - omega**2*1.0/eigs(j)*(-loadtim(k + 1) + tvals(i))*cos(-omega*(&
      -loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*&
      eigs(j)*(-loadtim(k + 1) + tvals(i)))*1.0/(-dT*eigs(j)*loadtim(k&
      + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k +&
      1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**&
      (-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT&
      **3)*loadmag(k)/dT + omega**2*1.0/eigs(j)*cos(-omega*(-loadtim(k&
      + 1) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(&
      -loadtim(k + 1) + tvals(i)))*loadtim(k)*1.0/(-dT*eigs(j)*loadtim(&
      k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k&
      + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)&
      **(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/&
      dT**3)*loadmag(k + 1)/dT - omega**2*1.0/eigs(j)*loadtim(k + 1)*&
      cos(-omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase)&
      *exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*loadmag(k)/dT - omega**2*tvals(i)*1.0/eigs(j)*&
      cos(-omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase)&
      *exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*loadmag(k + 1)/dT + omega**2*tvals(i)*1.0/eigs(&
      j)*cos(-omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(i) +&
      phase)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*loadmag(k)/dT + 2*omega*1.0/eigs(j)*exp(&
      -dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*sin(-omega*(-loadtim(k&
      + 1) + tvals(i)) + omega*tvals(i) + phase)*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*loadmag(k + 1)/dT - 2*omega*1.0/eigs(j)*exp(-dT&
      *eigs(j)*(-loadtim(k + 1) + tvals(i)))*sin(-omega*(-loadtim(k + 1&
      ) + tvals(i)) + omega*tvals(i) + phase)*1.0/(-dT*eigs(j)*loadtim(&
      k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k&
      + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)&
      **(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/&
      dT**3)*loadmag(k)/dT + omega**3*eigs(j)**(-2)*(-loadtim(k + 1) +&
      tvals(i))*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*sin(&
      -omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase)*1.0&
      /(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2&
      *1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k&
      )/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*&
      eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1)/dT**2 - omega**3*&
      eigs(j)**(-2)*(-loadtim(k + 1) + tvals(i))*exp(-dT*eigs(j)*(&
      -loadtim(k + 1) + tvals(i)))*sin(-omega*(-loadtim(k + 1) + tvals(&
      i)) + omega*tvals(i) + phase)*1.0/(-dT*eigs(j)*loadtim(k + 1) +&
      dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT&
      + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      loadmag(k)/dT**2 + omega**3*eigs(j)**(-2)*exp(-dT*eigs(j)*(&
      -loadtim(k + 1) + tvals(i)))*sin(-omega*(-loadtim(k + 1) + tvals(&
      i)) + omega*tvals(i) + phase)*loadtim(k)*1.0/(-dT*eigs(j)*loadtim&
      (k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(&
      k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(&
      j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)&
      /dT**3)*loadmag(k + 1)/dT**2 - omega**3*eigs(j)**(-2)*loadtim(k +&
      1)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*sin(-omega*(&
      -loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase)*1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*loadmag(k)/dT**2 - omega**3*tvals(i)*&
      eigs(j)**(-2)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*sin(&
      -omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase)*1.0&
      /(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2&
      *1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k&
      )/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*&
      eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1)/dT**2 + omega**3*&
      tvals(i)*eigs(j)**(-2)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i&
      )))*sin(-omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(i) +&
      phase)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) -&
      2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)&
      *loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k)/dT**2 - omega&
      **2*eigs(j)**(-2)*cos(-omega*(-loadtim(k + 1) + tvals(i)) + omega&
      *tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*&
      1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega&
      **2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1)/dT**2 +&
      omega**2*eigs(j)**(-2)*cos(-omega*(-loadtim(k + 1) + tvals(i)) +&
      omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(&
      i)))*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*&
      omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k)/dT**2)/dT -&
      1.0/eigs(j)*(dT*eigs(j)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs&
      (j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*&
      omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*(&
      -loadtim(k) + tvals(i))*cos(-omega*(-loadtim(k) + tvals(i)) +&
      omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))&
      *loadmag(k + 1) - dT*eigs(j)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT&
      *eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT +&
      2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*(&
      -loadtim(k) + tvals(i))*cos(-omega*(-loadtim(k) + tvals(i)) +&
      omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))&
      *loadmag(k) - dT*eigs(j)*loadtim(k + 1)*1.0/(-dT*eigs(j)*loadtim(&
      k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k&
      + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)&
      **(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/&
      dT**3)*cos(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) +&
      phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k) + dT*&
      eigs(j)*loadtim(k)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*cos(-omega*(&
      -loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)&
      *(-loadtim(k) + tvals(i)))*loadmag(k + 1) - dT*tvals(i)*eigs(j)*&
      1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega&
      **2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*cos(-omega*(-loadtim(k)&
      + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k&
      ) + tvals(i)))*loadmag(k + 1) + dT*tvals(i)*eigs(j)*1.0/(-dT*eigs&
      (j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(&
      j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT -&
      omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(&
      -3)*loadtim(k)/dT**3)*cos(-omega*(-loadtim(k) + tvals(i)) + omega&
      *tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*&
      loadmag(k) + omega*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*(-loadtim(k)&
      + tvals(i))*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*&
      (-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*loadmag(k + 1&
      ) - omega*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k&
      ) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/&
      eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT&
      **3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*(-loadtim(k) +&
      tvals(i))*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(&
      -loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*loadmag(k) -&
      omega*loadtim(k + 1)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)&
      *loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega&
      **2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k&
      + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*exp(-dT*&
      eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals&
      (i)) + omega*tvals(i) + phase)*loadmag(k) + omega*loadtim(k)*1.0/&
      (-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*&
      1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)&
      /dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs&
      (j)**(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(&
      i)))*sin(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase&
      )*loadmag(k + 1) - omega*tvals(i)*1.0/(-dT*eigs(j)*loadtim(k + 1&
      ) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)&
      /dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(&
      -3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**&
      3)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim&
      (k) + tvals(i)) + omega*tvals(i) + phase)*loadmag(k + 1) + omega*&
      tvals(i)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k)&
      - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(&
      j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(j)*(&
      -loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals(i)) +&
      omega*tvals(i) + phase)*loadmag(k) + 1.0/(-dT*eigs(j)*loadtim(k +&
      1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1&
      )/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(&
      -3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**&
      3)*cos(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*&
      exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k + 1) - 1.0/(&
      -dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*&
      1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)&
      /dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs&
      (j)**(-3)*loadtim(k)/dT**3)*cos(-omega*(-loadtim(k) + tvals(i)) +&
      omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))&
      *loadmag(k) + omega**2*1.0/eigs(j)*1.0/(-dT*eigs(j)*loadtim(k + 1&
      ) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)&
      /dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(&
      -3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**&
      3)*(-loadtim(k) + tvals(i))*cos(-omega*(-loadtim(k) + tvals(i)) +&
      omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))&
      *loadmag(k + 1)/dT - omega**2*1.0/eigs(j)*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*(-loadtim(k) + tvals(i))*cos(-omega*(-loadtim(k&
      ) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim&
      (k) + tvals(i)))*loadmag(k)/dT - omega**2*1.0/eigs(j)*loadtim(k +&
      1)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*&
      omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*cos(-omega*(-loadtim(k)&
      + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k&
      ) + tvals(i)))*loadmag(k)/dT + omega**2*1.0/eigs(j)*loadtim(k)*&
      1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega&
      **2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*cos(-omega*(-loadtim(k)&
      + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k&
      ) + tvals(i)))*loadmag(k + 1)/dT - omega**2*tvals(i)*1.0/eigs(j)*&
      1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega&
      **2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*cos(-omega*(-loadtim(k)&
      + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k&
      ) + tvals(i)))*loadmag(k + 1)/dT + omega**2*tvals(i)*1.0/eigs(j)*&
      1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega&
      **2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*cos(-omega*(-loadtim(k)&
      + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k&
      ) + tvals(i)))*loadmag(k)/dT + 2*omega*1.0/eigs(j)*1.0/(-dT*eigs(&
      j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j&
      )*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT -&
      omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(&
      -3)*loadtim(k)/dT**3)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*&
      sin(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*&
      loadmag(k + 1)/dT - 2*omega*1.0/eigs(j)*1.0/(-dT*eigs(j)*loadtim(&
      k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k&
      + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)&
      **(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/&
      dT**3)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(&
      -loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*loadmag(k)/dT +&
      omega**3*eigs(j)**(-2)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(&
      j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*&
      omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*(&
      -loadtim(k) + tvals(i))*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))&
      *sin(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*&
      loadmag(k + 1)/dT**2 - omega**3*eigs(j)**(-2)*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*(-loadtim(k) + tvals(i))*exp(-dT*eigs(j)*(&
      -loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals(i)) +&
      omega*tvals(i) + phase)*loadmag(k)/dT**2 - omega**3*eigs(j)**(-2)&
      *loadtim(k + 1)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(&
      j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals(i))&
      + omega*tvals(i) + phase)*loadmag(k)/dT**2 + omega**3*eigs(j)**(&
      -2)*loadtim(k)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(&
      j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals(i))&
      + omega*tvals(i) + phase)*loadmag(k + 1)/dT**2 - omega**3*tvals(i&
      )*eigs(j)**(-2)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(&
      j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals(i))&
      + omega*tvals(i) + phase)*loadmag(k + 1)/dT**2 + omega**3*tvals(i&
      )*eigs(j)**(-2)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(&
      j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals(i))&
      + omega*tvals(i) + phase)*loadmag(k)/dT**2 - omega**2*eigs(j)**(&
      -2)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*&
      omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*cos(-omega*(-loadtim(k)&
      + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k&
      ) + tvals(i)))*loadmag(k + 1)/dT**2 + omega**2*eigs(j)**(-2)*1.0/&
      (-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*&
      1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)&
      /dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs&
      (j)**(-3)*loadtim(k)/dT**3)*cos(-omega*(-loadtim(k) + tvals(i)) +&
      omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))&
      *loadmag(k)/dT**2)/dT)
                END DO
              END IF
            ELSE
              !t is in the load step
              IF(ABS(loadmag(k) - loadmag(k + 1)) <= &
                    (ABS(loadmag(k) + loadmag(k + 1))*EPSILON)) THEN
                !constant load
                DO j=0, neig-1
      a(i, j) = a(i, j) + (1.0/eigs(j)*(cos(omega*tvals(i) + phase)*1.0/&
      (1 + omega**2*eigs(j)**(-2)/dT**2) + omega*sin(omega*tvals(i) +&
      phase)*1.0/eigs(j)*1.0/(1 + omega**2*eigs(j)**(-2)/dT**2)/dT)*&
      loadmag(k)/dT - 1.0/eigs(j)*(1.0/(1 + omega**2*eigs(j)**(-2)/dT**&
      2)*cos(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*&
      exp(-dT*eigs(j)*(-loadtim(k) + tvals(i))) + omega*1.0/eigs(j)*1.0&
      /(1 + omega**2*eigs(j)**(-2)/dT**2)*exp(-dT*eigs(j)*(-loadtim(k)&
      + tvals(i)))*sin(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i&
      ) + phase)/dT)*loadmag(k)/dT)
                END DO
              ELSE
                !ramp load
                DO j=0, neig-1
      a(i, j) = a(i, j) + (1.0/eigs(j)*(-dT*cos(omega*tvals(i) + phase)*&
      eigs(j)*loadtim(k + 1)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(&
      j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*&
      omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      loadmag(k) + dT*cos(omega*tvals(i) + phase)*eigs(j)*loadtim(k)*&
      1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega&
      **2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1) - dT*&
      tvals(i)*cos(omega*tvals(i) + phase)*eigs(j)*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*loadmag(k + 1) + dT*tvals(i)*cos(omega*tvals(i&
      ) + phase)*eigs(j)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k) -&
      omega*sin(omega*tvals(i) + phase)*loadtim(k + 1)*1.0/(-dT*eigs(j)&
      *loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*loadmag(k) + omega*sin(omega*tvals(i) + phase)*&
      loadtim(k)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k&
      ) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/&
      eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT&
      **3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1) -&
      omega*tvals(i)*sin(omega*tvals(i) + phase)*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*loadmag(k + 1) + omega*tvals(i)*sin(omega*tvals&
      (i) + phase)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim&
      (k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/&
      eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT&
      **3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k) + cos(&
      omega*tvals(i) + phase)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs&
      (j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*&
      omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      loadmag(k + 1) - cos(omega*tvals(i) + phase)*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*loadmag(k) - omega**2*cos(omega*tvals(i) +&
      phase)*1.0/eigs(j)*loadtim(k + 1)*1.0/(-dT*eigs(j)*loadtim(k + 1&
      ) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)&
      /dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(&
      -3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**&
      3)*loadmag(k)/dT + omega**2*cos(omega*tvals(i) + phase)*1.0/eigs(&
      j)*loadtim(k)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1&
      )/dT - omega**2*tvals(i)*cos(omega*tvals(i) + phase)*1.0/eigs(j)*&
      1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega&
      **2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1)/dT +&
      omega**2*tvals(i)*cos(omega*tvals(i) + phase)*1.0/eigs(j)*1.0/(&
      -dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*&
      1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)&
      /dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs&
      (j)**(-3)*loadtim(k)/dT**3)*loadmag(k)/dT + 2*omega*sin(omega*&
      tvals(i) + phase)*1.0/eigs(j)*1.0/(-dT*eigs(j)*loadtim(k + 1) +&
      dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT&
      + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      loadmag(k + 1)/dT - 2*omega*sin(omega*tvals(i) + phase)*1.0/eigs(&
      j)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*&
      omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k)/dT - omega**3&
      *sin(omega*tvals(i) + phase)*eigs(j)**(-2)*loadtim(k + 1)*1.0/(&
      -dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*&
      1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)&
      /dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs&
      (j)**(-3)*loadtim(k)/dT**3)*loadmag(k)/dT**2 + omega**3*sin(omega&
      *tvals(i) + phase)*eigs(j)**(-2)*loadtim(k)*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*loadmag(k + 1)/dT**2 - omega**3*tvals(i)*sin(&
      omega*tvals(i) + phase)*eigs(j)**(-2)*1.0/(-dT*eigs(j)*loadtim(k&
      + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k +&
      1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**&
      (-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT&
      **3)*loadmag(k + 1)/dT**2 + omega**3*tvals(i)*sin(omega*tvals(i)&
      + phase)*eigs(j)**(-2)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(&
      j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*&
      omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      loadmag(k)/dT**2 - omega**2*cos(omega*tvals(i) + phase)*eigs(j)**&
      (-2)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*&
      omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1)/dT**2 +&
      omega**2*cos(omega*tvals(i) + phase)*eigs(j)**(-2)*1.0/(-dT*eigs(&
      j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j&
      )*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT -&
      omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(&
      -3)*loadtim(k)/dT**3)*loadmag(k)/dT**2)/dT - 1.0/eigs(j)*(dT*eigs&
      (j)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*&
      omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*(-loadtim(k) + tvals(i))&
      *cos(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*&
      exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k + 1) - dT*&
      eigs(j)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) -&
      2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)&
      *loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*(-loadtim(k) + tvals(i))&
      *cos(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*&
      exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k) - dT*eigs(j)&
      *loadtim(k + 1)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*cos(-omega*(&
      -loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)&
      *(-loadtim(k) + tvals(i)))*loadmag(k) + dT*eigs(j)*loadtim(k)*1.0&
      /(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2&
      *1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k&
      )/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*&
      eigs(j)**(-3)*loadtim(k)/dT**3)*cos(-omega*(-loadtim(k) + tvals(i&
      )) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals&
      (i)))*loadmag(k + 1) - dT*tvals(i)*eigs(j)*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*cos(-omega*(-loadtim(k) + tvals(i)) + omega*&
      tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*&
      loadmag(k + 1) + dT*tvals(i)*eigs(j)*1.0/(-dT*eigs(j)*loadtim(k +&
      1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1&
      )/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(&
      -3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**&
      3)*cos(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*&
      exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k) + omega*1.0/&
      (-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*&
      1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)&
      /dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs&
      (j)**(-3)*loadtim(k)/dT**3)*(-loadtim(k) + tvals(i))*exp(-dT*eigs&
      (j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals(i&
      )) + omega*tvals(i) + phase)*loadmag(k + 1) - omega*1.0/(-dT*eigs&
      (j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(&
      j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT -&
      omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(&
      -3)*loadtim(k)/dT**3)*(-loadtim(k) + tvals(i))*exp(-dT*eigs(j)*(&
      -loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals(i)) +&
      omega*tvals(i) + phase)*loadmag(k) - omega*loadtim(k + 1)*1.0/(&
      -dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*&
      1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)&
      /dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs&
      (j)**(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(&
      i)))*sin(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase&
      )*loadmag(k) + omega*loadtim(k)*1.0/(-dT*eigs(j)*loadtim(k + 1) +&
      dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT&
      + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k&
      ) + tvals(i)) + omega*tvals(i) + phase)*loadmag(k + 1) - omega*&
      tvals(i)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k)&
      - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(&
      j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(j)*(&
      -loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals(i)) +&
      omega*tvals(i) + phase)*loadmag(k + 1) + omega*tvals(i)*1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i&
      )))*sin(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)&
      *loadmag(k) + 1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*cos(-omega*(&
      -loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)&
      *(-loadtim(k) + tvals(i)))*loadmag(k + 1) - 1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*cos(-omega*(-loadtim(k) + tvals(i)) + omega*&
      tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*&
      loadmag(k) + omega**2*1.0/eigs(j)*1.0/(-dT*eigs(j)*loadtim(k + 1&
      ) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)&
      /dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(&
      -3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**&
      3)*(-loadtim(k) + tvals(i))*cos(-omega*(-loadtim(k) + tvals(i)) +&
      omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))&
      *loadmag(k + 1)/dT - omega**2*1.0/eigs(j)*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*(-loadtim(k) + tvals(i))*cos(-omega*(-loadtim(k&
      ) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim&
      (k) + tvals(i)))*loadmag(k)/dT - omega**2*1.0/eigs(j)*loadtim(k +&
      1)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*&
      omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*cos(-omega*(-loadtim(k)&
      + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k&
      ) + tvals(i)))*loadmag(k)/dT + omega**2*1.0/eigs(j)*loadtim(k)*&
      1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega&
      **2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*cos(-omega*(-loadtim(k)&
      + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k&
      ) + tvals(i)))*loadmag(k + 1)/dT - omega**2*tvals(i)*1.0/eigs(j)*&
      1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega&
      **2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*cos(-omega*(-loadtim(k)&
      + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k&
      ) + tvals(i)))*loadmag(k + 1)/dT + omega**2*tvals(i)*1.0/eigs(j)*&
      1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega&
      **2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*cos(-omega*(-loadtim(k)&
      + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k&
      ) + tvals(i)))*loadmag(k)/dT + 2*omega*1.0/eigs(j)*1.0/(-dT*eigs(&
      j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j&
      )*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT -&
      omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(&
      -3)*loadtim(k)/dT**3)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*&
      sin(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*&
      loadmag(k + 1)/dT - 2*omega*1.0/eigs(j)*1.0/(-dT*eigs(j)*loadtim(&
      k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k&
      + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)&
      **(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/&
      dT**3)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(&
      -loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*loadmag(k)/dT +&
      omega**3*eigs(j)**(-2)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(&
      j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*&
      omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*(&
      -loadtim(k) + tvals(i))*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))&
      *sin(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*&
      loadmag(k + 1)/dT**2 - omega**3*eigs(j)**(-2)*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*(-loadtim(k) + tvals(i))*exp(-dT*eigs(j)*(&
      -loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals(i)) +&
      omega*tvals(i) + phase)*loadmag(k)/dT**2 - omega**3*eigs(j)**(-2)&
      *loadtim(k + 1)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(&
      j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals(i))&
      + omega*tvals(i) + phase)*loadmag(k)/dT**2 + omega**3*eigs(j)**(&
      -2)*loadtim(k)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(&
      j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals(i))&
      + omega*tvals(i) + phase)*loadmag(k + 1)/dT**2 - omega**3*tvals(i&
      )*eigs(j)**(-2)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(&
      j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals(i))&
      + omega*tvals(i) + phase)*loadmag(k + 1)/dT**2 + omega**3*tvals(i&
      )*eigs(j)**(-2)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(&
      j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals(i))&
      + omega*tvals(i) + phase)*loadmag(k)/dT**2 - omega**2*eigs(j)**(&
      -2)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*&
      omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*cos(-omega*(-loadtim(k)&
      + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k&
      ) + tvals(i)))*loadmag(k + 1)/dT**2 + omega**2*eigs(j)**(-2)*1.0/&
      (-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*&
      1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)&
      /dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs&
      (j)**(-3)*loadtim(k)/dT**3)*cos(-omega*(-loadtim(k) + tvals(i)) +&
      omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))&
      *loadmag(k)/dT**2)/dT)
                END DO
              END IF
            END IF
          END DO
        END DO

      END SUBROUTINE


      SUBROUTINE edload_coslinear(loadtim, loadmag, omega, phase,&
                                  eigs, tvals, dT, a, neig, nload, nt)
        USE types
        IMPLICIT NONE

        INTEGER, intent(in) :: neig
        INTEGER, intent(in) :: nload
        INTEGER, intent(in) :: nt
        REAL(DP), intent(in), dimension(0:nload-1) :: loadtim
        REAL(DP), intent(in), dimension(0:nload-1) :: loadmag
        REAL(DP), intent(in), dimension(0:neig-1) :: eigs
        REAL(DP), intent(in), dimension(0:nt-1) :: tvals
        REAL(DP), intent(in) :: dT
        REAL(DP), intent(in) :: omega
        REAL(DP), intent(in) :: phase
        REAL(DP), intent(out), dimension(0:nt-1, 0:neig-1) :: a
        INTEGER :: i , j, k
        REAL(DP):: EPSILON
        a=0.0D0
        EPSILON = 0.0000005D0
        DO i = 0, nt-1
          DO k = 0, nload-2

            IF (tvals(i) < loadtim(k)) EXIT !t is before load step

            IF (tvals(i) >= loadtim(k + 1)) THEN
              !t is after the load step
              IF(ABS(loadtim(k) - loadtim(k + 1)) <= &
                (ABS(loadtim(k) + loadtim(k + 1))*EPSILON)) THEN
                !step load
                DO j=0, neig-1
      a(i, j) = a(i, j) + (cos(omega*loadtim(k) + phase)*exp(-dT*eigs(j)&
      *(-loadtim(k) + tvals(i)))*(loadmag(k + 1) - loadmag(k)))
                END DO
              ELSEIF(ABS(loadmag(k) - loadmag(k + 1)) <= &
                    (ABS(loadmag(k) + loadmag(k + 1))*EPSILON)) THEN
                !constant load
                DO j=0, neig-1
      a(i, j) = a(i, j) + (-omega*1.0/eigs(j)*(1.0/(1 + omega**2*eigs(j)&
      **(-2)/dT**2)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*sin(&
      -omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase) -&
      omega*1.0/eigs(j)*1.0/(1 + omega**2*eigs(j)**(-2)/dT**2)*cos(&
      -omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase)*exp&
      (-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))/dT)*loadmag(k)/dT +&
      omega*1.0/eigs(j)*(1.0/(1 + omega**2*eigs(j)**(-2)/dT**2)*exp(-dT&
      *eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) +&
      tvals(i)) + omega*tvals(i) + phase) - omega*1.0/eigs(j)*1.0/(1 +&
      omega**2*eigs(j)**(-2)/dT**2)*cos(-omega*(-loadtim(k) + tvals(i&
      )) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals&
      (i)))/dT)*loadmag(k)/dT)
                END DO
              ELSE
                !ramp load
                DO j=0, neig-1
      a(i, j) = a(i, j) + (-omega*(-loadtim(k + 1) + tvals(i))*exp(-dT*&
      eigs(j)*(-loadtim(k + 1) + tvals(i)))*sin(-omega*(-loadtim(k + 1&
      ) + tvals(i)) + omega*tvals(i) + phase)*1.0/(-dT*eigs(j)*loadtim(&
      k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k&
      + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)&
      **(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/&
      dT**3)*loadmag(k + 1) + omega*(-loadtim(k + 1) + tvals(i))*exp(&
      -dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*sin(-omega*(-loadtim(k&
      + 1) + tvals(i)) + omega*tvals(i) + phase)*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*loadmag(k) + omega*1.0/(-dT*eigs(j)*loadtim(k +&
      1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1&
      )/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(&
      -3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**&
      3)*(-loadtim(k) + tvals(i))*exp(-dT*eigs(j)*(-loadtim(k) + tvals(&
      i)))*sin(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase&
      )*loadmag(k + 1) - omega*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*&
      eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2&
      *omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*(&
      -loadtim(k) + tvals(i))*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))&
      *sin(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*&
      loadmag(k) - omega*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*&
      sin(-omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase)&
      *loadtim(k)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(&
      k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/&
      eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT&
      **3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1) -&
      omega*loadtim(k + 1)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)&
      *loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega&
      **2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k&
      + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*exp(-dT*&
      eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals&
      (i)) + omega*tvals(i) + phase)*loadmag(k) + omega*loadtim(k + 1)*&
      exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*sin(-omega*(&
      -loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase)*1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*loadmag(k) + omega*loadtim(k)*1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i&
      )))*sin(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)&
      *loadmag(k + 1) - omega*tvals(i)*1.0/(-dT*eigs(j)*loadtim(k + 1)&
      + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/&
      dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3&
      )*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)&
      *exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k&
      ) + tvals(i)) + omega*tvals(i) + phase)*loadmag(k + 1) + omega*&
      tvals(i)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k)&
      - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(&
      j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(j)*(&
      -loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals(i)) +&
      omega*tvals(i) + phase)*loadmag(k) + omega*tvals(i)*exp(-dT*eigs(&
      j)*(-loadtim(k + 1) + tvals(i)))*sin(-omega*(-loadtim(k + 1) +&
      tvals(i)) + omega*tvals(i) + phase)*1.0/(-dT*eigs(j)*loadtim(k +&
      1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1&
      )/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(&
      -3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**&
      3)*loadmag(k + 1) - omega*tvals(i)*exp(-dT*eigs(j)*(-loadtim(k +&
      1) + tvals(i)))*sin(-omega*(-loadtim(k + 1) + tvals(i)) + omega*&
      tvals(i) + phase)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k) +&
      1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega&
      **2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*cos(-omega*(-loadtim(k)&
      + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k&
      ) + tvals(i)))*loadmag(k + 1) - 1.0/(-dT*eigs(j)*loadtim(k + 1) +&
      dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT&
      + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      cos(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*exp&
      (-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k) - cos(-omega*(&
      -loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*&
      eigs(j)*(-loadtim(k + 1) + tvals(i)))*1.0/(-dT*eigs(j)*loadtim(k&
      + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k +&
      1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**&
      (-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT&
      **3)*loadmag(k + 1) + cos(-omega*(-loadtim(k + 1) + tvals(i)) +&
      omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(&
      i)))*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*&
      omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k) + omega**2*&
      1.0/eigs(j)*(-loadtim(k + 1) + tvals(i))*cos(-omega*(-loadtim(k +&
      1) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(&
      -loadtim(k + 1) + tvals(i)))*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT&
      *eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT +&
      2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      loadmag(k + 1)/dT - omega**2*1.0/eigs(j)*(-loadtim(k + 1) + tvals&
      (i))*cos(-omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(i) +&
      phase)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*loadmag(k)/dT - omega**2*1.0/eigs(j)*1.0&
      /(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2&
      *1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k&
      )/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*&
      eigs(j)**(-3)*loadtim(k)/dT**3)*(-loadtim(k) + tvals(i))*cos(&
      -omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT&
      *eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k + 1)/dT + omega**2*&
      1.0/eigs(j)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(&
      k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/&
      eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT&
      **3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*(-loadtim(k) +&
      tvals(i))*cos(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) +&
      phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k)/dT +&
      omega**2*1.0/eigs(j)*cos(-omega*(-loadtim(k + 1) + tvals(i)) +&
      omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(&
      i)))*loadtim(k)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1&
      )/dT + omega**2*1.0/eigs(j)*loadtim(k + 1)*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*cos(-omega*(-loadtim(k) + tvals(i)) + omega*&
      tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*&
      loadmag(k)/dT - omega**2*1.0/eigs(j)*loadtim(k + 1)*cos(-omega*(&
      -loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*&
      eigs(j)*(-loadtim(k + 1) + tvals(i)))*1.0/(-dT*eigs(j)*loadtim(k&
      + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k +&
      1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**&
      (-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT&
      **3)*loadmag(k)/dT - omega**2*1.0/eigs(j)*loadtim(k)*1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*cos(-omega*(-loadtim(k) + tvals(i)) +&
      omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))&
      *loadmag(k + 1)/dT + omega**2*tvals(i)*1.0/eigs(j)*1.0/(-dT*eigs(&
      j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j&
      )*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT -&
      omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(&
      -3)*loadtim(k)/dT**3)*cos(-omega*(-loadtim(k) + tvals(i)) + omega&
      *tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*&
      loadmag(k + 1)/dT - omega**2*tvals(i)*1.0/eigs(j)*1.0/(-dT*eigs(j&
      )*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)&
      *loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*cos(-omega*(-loadtim(k) + tvals(i)) + omega*&
      tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*&
      loadmag(k)/dT - omega**2*tvals(i)*1.0/eigs(j)*cos(-omega*(&
      -loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*&
      eigs(j)*(-loadtim(k + 1) + tvals(i)))*1.0/(-dT*eigs(j)*loadtim(k&
      + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k +&
      1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**&
      (-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT&
      **3)*loadmag(k + 1)/dT + omega**2*tvals(i)*1.0/eigs(j)*cos(-omega&
      *(-loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*&
      eigs(j)*(-loadtim(k + 1) + tvals(i)))*1.0/(-dT*eigs(j)*loadtim(k&
      + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k +&
      1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**&
      (-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT&
      **3)*loadmag(k)/dT + 2*omega*1.0/eigs(j)*1.0/(-dT*eigs(j)*loadtim&
      (k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(&
      k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(&
      j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)&
      /dT**3)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(&
      -loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*loadmag(k + 1)/&
      dT - 2*omega*1.0/eigs(j)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*&
      eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2&
      *omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k&
      ) + tvals(i)) + omega*tvals(i) + phase)*loadmag(k)/dT - 2*omega*&
      1.0/eigs(j)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*sin(&
      -omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase)*1.0&
      /(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2&
      *1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k&
      )/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*&
      eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1)/dT + 2*omega*1.0/&
      eigs(j)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*sin(-omega*&
      (-loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase)*1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*loadmag(k)/dT - omega**3*eigs(j)**(-2)*(&
      -loadtim(k + 1) + tvals(i))*exp(-dT*eigs(j)*(-loadtim(k + 1) +&
      tvals(i)))*sin(-omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(&
      i) + phase)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(&
      k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/&
      eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT&
      **3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1)/dT&
      **2 + omega**3*eigs(j)**(-2)*(-loadtim(k + 1) + tvals(i))*exp(-dT&
      *eigs(j)*(-loadtim(k + 1) + tvals(i)))*sin(-omega*(-loadtim(k + 1&
      ) + tvals(i)) + omega*tvals(i) + phase)*1.0/(-dT*eigs(j)*loadtim(&
      k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k&
      + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)&
      **(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/&
      dT**3)*loadmag(k)/dT**2 + omega**3*eigs(j)**(-2)*1.0/(-dT*eigs(j)&
      *loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*(-loadtim(k) + tvals(i))*exp(-dT*eigs(j)*(&
      -loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals(i)) +&
      omega*tvals(i) + phase)*loadmag(k + 1)/dT**2 - omega**3*eigs(j)**&
      (-2)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*&
      omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*(-loadtim(k) + tvals(i))&
      *exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k&
      ) + tvals(i)) + omega*tvals(i) + phase)*loadmag(k)/dT**2 - omega&
      **3*eigs(j)**(-2)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*&
      sin(-omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase)&
      *loadtim(k)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(&
      k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/&
      eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT&
      **3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1)/dT&
      **2 - omega**3*eigs(j)**(-2)*loadtim(k + 1)*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(&
      -omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*loadmag&
      (k)/dT**2 + omega**3*eigs(j)**(-2)*loadtim(k + 1)*exp(-dT*eigs(j)&
      *(-loadtim(k + 1) + tvals(i)))*sin(-omega*(-loadtim(k + 1) +&
      tvals(i)) + omega*tvals(i) + phase)*1.0/(-dT*eigs(j)*loadtim(k +&
      1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1&
      )/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(&
      -3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**&
      3)*loadmag(k)/dT**2 + omega**3*eigs(j)**(-2)*loadtim(k)*1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i&
      )))*sin(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)&
      *loadmag(k + 1)/dT**2 - omega**3*tvals(i)*eigs(j)**(-2)*1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i&
      )))*sin(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)&
      *loadmag(k + 1)/dT**2 + omega**3*tvals(i)*eigs(j)**(-2)*1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i&
      )))*sin(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)&
      *loadmag(k)/dT**2 + omega**3*tvals(i)*eigs(j)**(-2)*exp(-dT*eigs(&
      j)*(-loadtim(k + 1) + tvals(i)))*sin(-omega*(-loadtim(k + 1) +&
      tvals(i)) + omega*tvals(i) + phase)*1.0/(-dT*eigs(j)*loadtim(k +&
      1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1&
      )/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(&
      -3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**&
      3)*loadmag(k + 1)/dT**2 - omega**3*tvals(i)*eigs(j)**(-2)*exp(-dT&
      *eigs(j)*(-loadtim(k + 1) + tvals(i)))*sin(-omega*(-loadtim(k + 1&
      ) + tvals(i)) + omega*tvals(i) + phase)*1.0/(-dT*eigs(j)*loadtim(&
      k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k&
      + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)&
      **(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/&
      dT**3)*loadmag(k)/dT**2 - omega**2*eigs(j)**(-2)*1.0/(-dT*eigs(j)&
      *loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*cos(-omega*(-loadtim(k) + tvals(i)) + omega*&
      tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*&
      loadmag(k + 1)/dT**2 + omega**2*eigs(j)**(-2)*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*cos(-omega*(-loadtim(k) + tvals(i)) + omega*&
      tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*&
      loadmag(k)/dT**2 + omega**2*eigs(j)**(-2)*cos(-omega*(-loadtim(k&
      + 1) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(&
      -loadtim(k + 1) + tvals(i)))*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT&
      *eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT +&
      2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      loadmag(k + 1)/dT**2 - omega**2*eigs(j)**(-2)*cos(-omega*(&
      -loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*&
      eigs(j)*(-loadtim(k + 1) + tvals(i)))*1.0/(-dT*eigs(j)*loadtim(k&
      + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k +&
      1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**&
      (-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT&
      **3)*loadmag(k)/dT**2 + omega**4*eigs(j)**(-3)*(-loadtim(k + 1) +&
      tvals(i))*cos(-omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(i&
      ) + phase)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*1.0/(-dT&
      *eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*loadmag(k + 1)/dT**3 - omega**4*eigs(j)&
      **(-3)*(-loadtim(k + 1) + tvals(i))*cos(-omega*(-loadtim(k + 1) +&
      tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k +&
      1) + tvals(i)))*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k)/dT&
      **3 - omega**4*eigs(j)**(-3)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT&
      *eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT +&
      2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*(&
      -loadtim(k) + tvals(i))*cos(-omega*(-loadtim(k) + tvals(i)) +&
      omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))&
      *loadmag(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*(-loadtim(k) + tvals(i))*cos(-omega*(-loadtim(k&
      ) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim&
      (k) + tvals(i)))*loadmag(k)/dT**3 + omega**4*eigs(j)**(-3)*cos(&
      -omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase)*exp&
      (-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*loadtim(k)*1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*loadmag(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k + 1)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j&
      )*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega&
      **2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k&
      + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*cos(-omega*&
      (-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j&
      )*(-loadtim(k) + tvals(i)))*loadmag(k)/dT**3 - omega**4*eigs(j)**&
      (-3)*loadtim(k + 1)*cos(-omega*(-loadtim(k + 1) + tvals(i)) +&
      omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(&
      i)))*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*&
      omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k)/dT**3 - omega&
      **4*eigs(j)**(-3)*loadtim(k)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT&
      *eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT +&
      2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      cos(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*exp&
      (-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k + 1)/dT**3 +&
      omega**4*tvals(i)*eigs(j)**(-3)*1.0/(-dT*eigs(j)*loadtim(k + 1) +&
      dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT&
      + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      cos(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*exp&
      (-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k + 1)/dT**3 -&
      omega**4*tvals(i)*eigs(j)**(-3)*1.0/(-dT*eigs(j)*loadtim(k + 1) +&
      dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT&
      + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      cos(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*exp&
      (-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k)/dT**3 - omega**&
      4*tvals(i)*eigs(j)**(-3)*cos(-omega*(-loadtim(k + 1) + tvals(i))&
      + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k + 1) +&
      tvals(i)))*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k&
      ) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/&
      eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT&
      **3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1)/dT&
      **3 + omega**4*tvals(i)*eigs(j)**(-3)*cos(-omega*(-loadtim(k + 1&
      ) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim&
      (k + 1) + tvals(i)))*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)&
      *loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega&
      **2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k&
      + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k)/&
      dT**3)
                END DO
              END IF
            ELSE
              !t is in the load step
              IF(ABS(loadmag(k) - loadmag(k + 1)) <= &
                    (ABS(loadmag(k) + loadmag(k + 1))*EPSILON)) THEN
                !constant load
                DO j=0, neig-1
      a(i, j) = a(i, j) + (-omega*1.0/eigs(j)*(sin(omega*tvals(i) +&
      phase)*1.0/(1 + omega**2*eigs(j)**(-2)/dT**2) - omega*cos(omega*&
      tvals(i) + phase)*1.0/eigs(j)*1.0/(1 + omega**2*eigs(j)**(-2)/dT&
      **2)/dT)*loadmag(k)/dT + omega*1.0/eigs(j)*(1.0/(1 + omega**2*&
      eigs(j)**(-2)/dT**2)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*&
      sin(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase) -&
      omega*1.0/eigs(j)*1.0/(1 + omega**2*eigs(j)**(-2)/dT**2)*cos(&
      -omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT&
      *eigs(j)*(-loadtim(k) + tvals(i)))/dT)*loadmag(k)/dT)
                END DO
              ELSE
                !ramp load
                DO j=0, neig-1
      a(i, j) = a(i, j) + (omega*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*&
      eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2&
      *omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*(&
      -loadtim(k) + tvals(i))*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))&
      *sin(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*&
      loadmag(k + 1) - omega*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(&
      j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*&
      omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*(&
      -loadtim(k) + tvals(i))*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))&
      *sin(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*&
      loadmag(k) + omega*sin(omega*tvals(i) + phase)*loadtim(k + 1)*1.0&
      /(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2&
      *1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k&
      )/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*&
      eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k) - omega*sin(omega*&
      tvals(i) + phase)*loadtim(k)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT&
      *eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT +&
      2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      loadmag(k + 1) - omega*loadtim(k + 1)*1.0/(-dT*eigs(j)*loadtim(k&
      + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k +&
      1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**&
      (-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT&
      **3)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(&
      -loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*loadmag(k) +&
      omega*loadtim(k)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(&
      j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals(i))&
      + omega*tvals(i) + phase)*loadmag(k + 1) - omega*tvals(i)*1.0/(&
      -dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*&
      1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)&
      /dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs&
      (j)**(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(&
      i)))*sin(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase&
      )*loadmag(k + 1) + omega*tvals(i)*1.0/(-dT*eigs(j)*loadtim(k + 1&
      ) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)&
      /dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(&
      -3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**&
      3)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim&
      (k) + tvals(i)) + omega*tvals(i) + phase)*loadmag(k) + omega*&
      tvals(i)*sin(omega*tvals(i) + phase)*1.0/(-dT*eigs(j)*loadtim(k +&
      1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1&
      )/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(&
      -3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**&
      3)*loadmag(k + 1) - omega*tvals(i)*sin(omega*tvals(i) + phase)*&
      1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega&
      **2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k) + 1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*cos(-omega*(-loadtim(k) + tvals(i)) +&
      omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))&
      *loadmag(k + 1) - 1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*cos(-omega*(&
      -loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)&
      *(-loadtim(k) + tvals(i)))*loadmag(k) - cos(omega*tvals(i) +&
      phase)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) -&
      2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)&
      *loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1) + cos(&
      omega*tvals(i) + phase)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs&
      (j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*&
      omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      loadmag(k) - omega**2*cos(omega*tvals(i) + phase)*1.0/eigs(j)*&
      loadtim(k + 1)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k)/dT&
      + omega**2*cos(omega*tvals(i) + phase)*1.0/eigs(j)*loadtim(k)*1.0&
      /(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2&
      *1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k&
      )/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*&
      eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1)/dT - omega**2*1.0/&
      eigs(j)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) -&
      2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)&
      *loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*(-loadtim(k) + tvals(i))&
      *cos(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*&
      exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k + 1)/dT +&
      omega**2*1.0/eigs(j)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)&
      *loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega&
      **2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k&
      + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*(-loadtim(k&
      ) + tvals(i))*cos(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i&
      ) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k)/&
      dT + omega**2*1.0/eigs(j)*loadtim(k + 1)*1.0/(-dT*eigs(j)*loadtim&
      (k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(&
      k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(&
      j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)&
      /dT**3)*cos(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) +&
      phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k)/dT -&
      omega**2*1.0/eigs(j)*loadtim(k)*1.0/(-dT*eigs(j)*loadtim(k + 1) +&
      dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT&
      + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      cos(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*exp&
      (-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k + 1)/dT - omega&
      **2*tvals(i)*cos(omega*tvals(i) + phase)*1.0/eigs(j)*1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*loadmag(k + 1)/dT + omega**2*tvals(i)*&
      cos(omega*tvals(i) + phase)*1.0/eigs(j)*1.0/(-dT*eigs(j)*loadtim(&
      k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k&
      + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)&
      **(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/&
      dT**3)*loadmag(k)/dT + omega**2*tvals(i)*1.0/eigs(j)*1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*cos(-omega*(-loadtim(k) + tvals(i)) +&
      omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))&
      *loadmag(k + 1)/dT - omega**2*tvals(i)*1.0/eigs(j)*1.0/(-dT*eigs(&
      j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j&
      )*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT -&
      omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(&
      -3)*loadtim(k)/dT**3)*cos(-omega*(-loadtim(k) + tvals(i)) + omega&
      *tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*&
      loadmag(k)/dT - 2*omega*sin(omega*tvals(i) + phase)*1.0/eigs(j)*&
      1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega&
      **2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1)/dT + 2*&
      omega*sin(omega*tvals(i) + phase)*1.0/eigs(j)*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*loadmag(k)/dT + 2*omega*1.0/eigs(j)*1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i&
      )))*sin(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)&
      *loadmag(k + 1)/dT - 2*omega*1.0/eigs(j)*1.0/(-dT*eigs(j)*loadtim&
      (k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(&
      k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(&
      j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)&
      /dT**3)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(&
      -loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*loadmag(k)/dT +&
      omega**3*sin(omega*tvals(i) + phase)*eigs(j)**(-2)*loadtim(k + 1)&
      *1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*&
      omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k)/dT**2 - omega&
      **3*sin(omega*tvals(i) + phase)*eigs(j)**(-2)*loadtim(k)*1.0/(-dT&
      *eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*loadmag(k + 1)/dT**2 + omega**3*eigs(j)&
      **(-2)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) -&
      2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)&
      *loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*(-loadtim(k) + tvals(i))&
      *exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k&
      ) + tvals(i)) + omega*tvals(i) + phase)*loadmag(k + 1)/dT**2 -&
      omega**3*eigs(j)**(-2)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(&
      j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*&
      omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*(&
      -loadtim(k) + tvals(i))*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))&
      *sin(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*&
      loadmag(k)/dT**2 - omega**3*eigs(j)**(-2)*loadtim(k + 1)*1.0/(-dT&
      *eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i&
      )))*sin(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)&
      *loadmag(k)/dT**2 + omega**3*eigs(j)**(-2)*loadtim(k)*1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i&
      )))*sin(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)&
      *loadmag(k + 1)/dT**2 + omega**3*tvals(i)*sin(omega*tvals(i) +&
      phase)*eigs(j)**(-2)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)&
      *loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega&
      **2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k&
      + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k +&
      1)/dT**2 - omega**3*tvals(i)*sin(omega*tvals(i) + phase)*eigs(j)&
      **(-2)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) -&
      2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)&
      *loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k)/dT**2 - omega&
      **3*tvals(i)*eigs(j)**(-2)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*&
      eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2&
      *omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k&
      ) + tvals(i)) + omega*tvals(i) + phase)*loadmag(k + 1)/dT**2 +&
      omega**3*tvals(i)*eigs(j)**(-2)*1.0/(-dT*eigs(j)*loadtim(k + 1) +&
      dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT&
      + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k&
      ) + tvals(i)) + omega*tvals(i) + phase)*loadmag(k)/dT**2 + omega&
      **2*cos(omega*tvals(i) + phase)*eigs(j)**(-2)*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*loadmag(k + 1)/dT**2 - omega**2*cos(omega*tvals&
      (i) + phase)*eigs(j)**(-2)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*&
      eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2&
      *omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      loadmag(k)/dT**2 - omega**2*eigs(j)**(-2)*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*cos(-omega*(-loadtim(k) + tvals(i)) + omega*&
      tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*&
      loadmag(k + 1)/dT**2 + omega**2*eigs(j)**(-2)*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*cos(-omega*(-loadtim(k) + tvals(i)) + omega*&
      tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*&
      loadmag(k)/dT**2 - omega**4*cos(omega*tvals(i) + phase)*eigs(j)**&
      (-3)*loadtim(k + 1)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k)/dT&
      **3 + omega**4*cos(omega*tvals(i) + phase)*eigs(j)**(-3)*loadtim(&
      k)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*&
      omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1)/dT**3 -&
      omega**4*eigs(j)**(-3)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(&
      j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*&
      omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*(&
      -loadtim(k) + tvals(i))*cos(-omega*(-loadtim(k) + tvals(i)) +&
      omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))&
      *loadmag(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*(-loadtim(k) + tvals(i))*cos(-omega*(-loadtim(k&
      ) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim&
      (k) + tvals(i)))*loadmag(k)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*cos(-omega*(&
      -loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)&
      *(-loadtim(k) + tvals(i)))*loadmag(k)/dT**3 - omega**4*eigs(j)**(&
      -3)*loadtim(k)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*cos(-omega*(&
      -loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)&
      *(-loadtim(k) + tvals(i)))*loadmag(k + 1)/dT**3 - omega**4*tvals(&
      i)*cos(omega*tvals(i) + phase)*eigs(j)**(-3)*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*loadmag(k + 1)/dT**3 + omega**4*tvals(i)*cos(&
      omega*tvals(i) + phase)*eigs(j)**(-3)*1.0/(-dT*eigs(j)*loadtim(k&
      + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k +&
      1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**&
      (-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT&
      **3)*loadmag(k)/dT**3 + omega**4*tvals(i)*eigs(j)**(-3)*1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*cos(-omega*(-loadtim(k) + tvals(i)) +&
      omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))&
      *loadmag(k + 1)/dT**3 - omega**4*tvals(i)*eigs(j)**(-3)*1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*cos(-omega*(-loadtim(k) + tvals(i)) +&
      omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))&
      *loadmag(k)/dT**3)
                END DO
              END IF
            END IF
          END DO
        END DO

      END SUBROUTINE



      SUBROUTINE dim1sin_dd_abddf_linear(m, at, ab, bt, bb, zt, zb, a, &
                                    neig, nlayers)
        USE types
        IMPLICIT NONE

        INTEGER, intent(in) :: neig
        INTEGER, intent(in) :: nlayers
        REAL(DP), intent(in), dimension(0:neig-1) ::m
        REAL(DP), intent(in), dimension(0:nlayers-1) :: at,ab,bt,bb,zt,zb
        REAL(DP), intent(out), dimension(0:neig-1, 0:neig-1) :: a
        INTEGER :: i , j, layer
        REAL(DP) :: a_slope, b_slope


        a=0.0D0
        DO layer = 0, nlayers-1
          a_slope = (ab(layer) - at(layer)) / (zb(layer) - zt(layer))
          b_slope = (bb(layer) - bt(layer)) / (zb(layer) - zt(layer))
          DO j = 0, neig-1
              i=j
      a(i, i) = a(i, i) + (-m(i)**4*((1.0d0/4.0d0)*a_slope*b_slope*m(i)&
      **(-3)*cos(zt(layer)*m(i))*sin(zt(layer)*m(i)) + (1.0d0/4.0d0)*&
      a_slope*b_slope*zt(layer)*m(i)**(-2)*sin(zt(layer)*m(i))**2 + (&
      1.0d0/4.0d0)*a_slope*b_slope*zt(layer)*m(i)**(-2)*cos(zt(layer)*m&
      (i))**2 + (1.0d0/6.0d0)*a_slope*b_slope*zt(layer)**3*sin(zt(layer&
      )*m(i))**2 + (1.0d0/6.0d0)*a_slope*b_slope*zt(layer)**3*cos(zt(&
      layer)*m(i))**2 - 1.0d0/4.0d0*a_slope*bt(layer)*m(i)**(-2)*cos(zt&
      (layer)*m(i))**2 - 1.0d0/4.0d0*a_slope*zt(layer)**2*bt(layer)*sin&
      (zt(layer)*m(i))**2 - 1.0d0/4.0d0*a_slope*zt(layer)**2*bt(layer)*&
      cos(zt(layer)*m(i))**2 - 1.0d0/4.0d0*b_slope*at(layer)*m(i)**(-2)&
      *cos(zt(layer)*m(i))**2 - 1.0d0/4.0d0*b_slope*zt(layer)**2*at(&
      layer)*sin(zt(layer)*m(i))**2 - 1.0d0/4.0d0*b_slope*zt(layer)**2*&
      at(layer)*cos(zt(layer)*m(i))**2 - 1.0d0/2.0d0*bt(layer)*at(layer&
      )*1.0/m(i)*cos(zt(layer)*m(i))*sin(zt(layer)*m(i)) + (1.0d0/2.0d0&
      )*zt(layer)*bt(layer)*at(layer)*sin(zt(layer)*m(i))**2 + (1.0d0/&
      2.0d0)*zt(layer)*bt(layer)*at(layer)*cos(zt(layer)*m(i))**2) + m(&
      i)**4*((1.0d0/4.0d0)*a_slope*b_slope*m(i)**(-3)*cos(zb(layer)*m(i&
      ))*sin(zb(layer)*m(i)) + (1.0d0/4.0d0)*a_slope*b_slope*zb(layer)*&
      m(i)**(-2)*sin(zb(layer)*m(i))**2 - 1.0d0/4.0d0*a_slope*b_slope*&
      zb(layer)*m(i)**(-2)*cos(zb(layer)*m(i))**2 + a_slope*b_slope*zb(&
      layer)*zt(layer)*1.0/m(i)*cos(zb(layer)*m(i))*sin(zb(layer)*m(i&
      )) + (1.0d0/2.0d0)*a_slope*b_slope*zb(layer)*zt(layer)**2*sin(zb(&
      layer)*m(i))**2 + (1.0d0/2.0d0)*a_slope*b_slope*zb(layer)*zt(&
      layer)**2*cos(zb(layer)*m(i))**2 - 1.0d0/2.0d0*a_slope*b_slope*zb&
      (layer)**2*1.0/m(i)*cos(zb(layer)*m(i))*sin(zb(layer)*m(i)) -&
      1.0d0/2.0d0*a_slope*b_slope*zb(layer)**2*zt(layer)*sin(zb(layer)*&
      m(i))**2 - 1.0d0/2.0d0*a_slope*b_slope*zb(layer)**2*zt(layer)*cos&
      (zb(layer)*m(i))**2 + (1.0d0/6.0d0)*a_slope*b_slope*zb(layer)**3*&
      sin(zb(layer)*m(i))**2 + (1.0d0/6.0d0)*a_slope*b_slope*zb(layer)&
      **3*cos(zb(layer)*m(i))**2 + (1.0d0/2.0d0)*a_slope*b_slope*zt(&
      layer)*m(i)**(-2)*cos(zb(layer)*m(i))**2 - 1.0d0/2.0d0*a_slope*&
      b_slope*zt(layer)**2*1.0/m(i)*cos(zb(layer)*m(i))*sin(zb(layer)*m&
      (i)) - 1.0d0/4.0d0*a_slope*bt(layer)*m(i)**(-2)*cos(zb(layer)*m(i&
      ))**2 - 1.0d0/2.0d0*a_slope*zb(layer)*bt(layer)*1.0/m(i)*cos(zb(&
      layer)*m(i))*sin(zb(layer)*m(i)) - 1.0d0/2.0d0*a_slope*zb(layer)*&
      zt(layer)*bt(layer)*sin(zb(layer)*m(i))**2 - 1.0d0/2.0d0*a_slope*&
      zb(layer)*zt(layer)*bt(layer)*cos(zb(layer)*m(i))**2 + (1.0d0/&
      4.0d0)*a_slope*zb(layer)**2*bt(layer)*sin(zb(layer)*m(i))**2 + (&
      1.0d0/4.0d0)*a_slope*zb(layer)**2*bt(layer)*cos(zb(layer)*m(i))**&
      2 + (1.0d0/2.0d0)*a_slope*zt(layer)*bt(layer)*1.0/m(i)*cos(zb(&
      layer)*m(i))*sin(zb(layer)*m(i)) - 1.0d0/4.0d0*b_slope*at(layer)*&
      m(i)**(-2)*cos(zb(layer)*m(i))**2 - 1.0d0/2.0d0*b_slope*zb(layer)&
      *at(layer)*1.0/m(i)*cos(zb(layer)*m(i))*sin(zb(layer)*m(i)) -&
      1.0d0/2.0d0*b_slope*zb(layer)*zt(layer)*at(layer)*sin(zb(layer)*m&
      (i))**2 - 1.0d0/2.0d0*b_slope*zb(layer)*zt(layer)*at(layer)*cos(&
      zb(layer)*m(i))**2 + (1.0d0/4.0d0)*b_slope*zb(layer)**2*at(layer)&
      *sin(zb(layer)*m(i))**2 + (1.0d0/4.0d0)*b_slope*zb(layer)**2*at(&
      layer)*cos(zb(layer)*m(i))**2 + (1.0d0/2.0d0)*b_slope*zt(layer)*&
      at(layer)*1.0/m(i)*cos(zb(layer)*m(i))*sin(zb(layer)*m(i)) -&
      1.0d0/2.0d0*bt(layer)*at(layer)*1.0/m(i)*cos(zb(layer)*m(i))*sin(&
      zb(layer)*m(i)) + (1.0d0/2.0d0)*zb(layer)*bt(layer)*at(layer)*sin&
      (zb(layer)*m(i))**2 + (1.0d0/2.0d0)*zb(layer)*bt(layer)*at(layer)&
      *cos(zb(layer)*m(i))**2))
            DO i = j+1, neig-1
      a(i, j) = a(i, j) + (-m(j)**2*m(i)**2*(2*a_slope*b_slope*sin(zt(&
      layer)*m(j))*m(i)**3*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4&
      *m(i)**2 - m(j)**6)*cos(zt(layer)*m(i)) - 6*a_slope*b_slope*m(j)*&
      cos(zt(layer)*m(j))*m(i)**2*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*&
      m(j)**4*m(i)**2 - m(j)**6)*sin(zt(layer)*m(i)) + 6*a_slope*&
      b_slope*m(j)**2*sin(zt(layer)*m(j))*m(i)*1.0/(m(i)**6 - 3*m(j)**2&
      *m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(zt(layer)*m(i)) - 2*&
      a_slope*b_slope*m(j)**3*cos(zt(layer)*m(j))*1.0/(m(i)**6 - 3*m(j)&
      **2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*sin(zt(layer)*m(i)) +&
      a_slope*bt(layer)*sin(zt(layer)*m(j))*m(i)**4*1.0/(m(i)**6 - 3*m(&
      j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*sin(zt(layer)*m(i))&
      + 2*a_slope*bt(layer)*m(j)*cos(zt(layer)*m(j))*m(i)**3*1.0/(m(i)&
      **6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(zt(&
      layer)*m(i)) - 2*a_slope*bt(layer)*m(j)**3*cos(zt(layer)*m(j))*m(&
      i)*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6&
      )*cos(zt(layer)*m(i)) - a_slope*bt(layer)*m(j)**4*sin(zt(layer)*m&
      (j))*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)&
      **6)*sin(zt(layer)*m(i)) + b_slope*at(layer)*sin(zt(layer)*m(j))*&
      m(i)**4*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(&
      j)**6)*sin(zt(layer)*m(i)) + 2*b_slope*at(layer)*m(j)*cos(zt(&
      layer)*m(j))*m(i)**3*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4&
      *m(i)**2 - m(j)**6)*cos(zt(layer)*m(i)) - 2*b_slope*at(layer)*m(j&
      )**3*cos(zt(layer)*m(j))*m(i)*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 +&
      3*m(j)**4*m(i)**2 - m(j)**6)*cos(zt(layer)*m(i)) - b_slope*at(&
      layer)*m(j)**4*sin(zt(layer)*m(j))*1.0/(m(i)**6 - 3*m(j)**2*m(i)&
      **4 + 3*m(j)**4*m(i)**2 - m(j)**6)*sin(zt(layer)*m(i)) - bt(layer&
      )*at(layer)*sin(zt(layer)*m(j))*m(i)**5*1.0/(m(i)**6 - 3*m(j)**2*&
      m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(zt(layer)*m(i)) + bt(&
      layer)*at(layer)*m(j)*cos(zt(layer)*m(j))*m(i)**4*1.0/(m(i)**6 -&
      3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*sin(zt(layer)*m(&
      i)) + 2*bt(layer)*at(layer)*m(j)**2*sin(zt(layer)*m(j))*m(i)**3*&
      1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*&
      cos(zt(layer)*m(i)) - 2*bt(layer)*at(layer)*m(j)**3*cos(zt(layer)&
      *m(j))*m(i)**2*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)&
      **2 - m(j)**6)*sin(zt(layer)*m(i)) - bt(layer)*at(layer)*m(j)**4*&
      sin(zt(layer)*m(j))*m(i)*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j&
      )**4*m(i)**2 - m(j)**6)*cos(zt(layer)*m(i)) + bt(layer)*at(layer)&
      *m(j)**5*cos(zt(layer)*m(j))*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3&
      *m(j)**4*m(i)**2 - m(j)**6)*sin(zt(layer)*m(i))) + m(j)**2*m(i)**&
      2*(2*a_slope*b_slope*sin(zb(layer)*m(j))*m(i)**3*1.0/(m(i)**6 - 3&
      *m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(zb(layer)*m(i&
      )) - 6*a_slope*b_slope*m(j)*cos(zb(layer)*m(j))*m(i)**2*1.0/(m(i)&
      **6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*sin(zb(&
      layer)*m(i)) + 6*a_slope*b_slope*m(j)**2*sin(zb(layer)*m(j))*m(i)&
      *1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*&
      cos(zb(layer)*m(i)) - 2*a_slope*b_slope*m(j)**3*cos(zb(layer)*m(j&
      ))*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6&
      )*sin(zb(layer)*m(i)) + 2*a_slope*b_slope*zb(layer)*sin(zb(layer)&
      *m(j))*m(i)**4*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)&
      **2 - m(j)**6)*sin(zb(layer)*m(i)) + 4*a_slope*b_slope*zb(layer)*&
      m(j)*cos(zb(layer)*m(j))*m(i)**3*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4&
      + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(zb(layer)*m(i)) - 4*a_slope*&
      b_slope*zb(layer)*m(j)**3*cos(zb(layer)*m(j))*m(i)*1.0/(m(i)**6 -&
      3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(zb(layer)*m(&
      i)) - 2*a_slope*b_slope*zb(layer)*m(j)**4*sin(zb(layer)*m(j))*1.0&
      /(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*sin(&
      zb(layer)*m(i)) + 2*a_slope*b_slope*zb(layer)*zt(layer)*sin(zb(&
      layer)*m(j))*m(i)**5*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4&
      *m(i)**2 - m(j)**6)*cos(zb(layer)*m(i)) - 2*a_slope*b_slope*zb(&
      layer)*zt(layer)*m(j)*cos(zb(layer)*m(j))*m(i)**4*1.0/(m(i)**6 -&
      3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*sin(zb(layer)*m(&
      i)) - 4*a_slope*b_slope*zb(layer)*zt(layer)*m(j)**2*sin(zb(layer)&
      *m(j))*m(i)**3*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)&
      **2 - m(j)**6)*cos(zb(layer)*m(i)) + 4*a_slope*b_slope*zb(layer)*&
      zt(layer)*m(j)**3*cos(zb(layer)*m(j))*m(i)**2*1.0/(m(i)**6 - 3*m(&
      j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*sin(zb(layer)*m(i))&
      + 2*a_slope*b_slope*zb(layer)*zt(layer)*m(j)**4*sin(zb(layer)*m(j&
      ))*m(i)*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(&
      j)**6)*cos(zb(layer)*m(i)) - 2*a_slope*b_slope*zb(layer)*zt(layer&
      )*m(j)**5*cos(zb(layer)*m(j))*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 +&
      3*m(j)**4*m(i)**2 - m(j)**6)*sin(zb(layer)*m(i)) - a_slope*&
      b_slope*zb(layer)**2*sin(zb(layer)*m(j))*m(i)**5*1.0/(m(i)**6 - 3&
      *m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(zb(layer)*m(i&
      )) + a_slope*b_slope*zb(layer)**2*m(j)*cos(zb(layer)*m(j))*m(i)**&
      4*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)&
      *sin(zb(layer)*m(i)) + 2*a_slope*b_slope*zb(layer)**2*m(j)**2*sin&
      (zb(layer)*m(j))*m(i)**3*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j&
      )**4*m(i)**2 - m(j)**6)*cos(zb(layer)*m(i)) - 2*a_slope*b_slope*&
      zb(layer)**2*m(j)**3*cos(zb(layer)*m(j))*m(i)**2*1.0/(m(i)**6 - 3&
      *m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*sin(zb(layer)*m(i&
      )) - a_slope*b_slope*zb(layer)**2*m(j)**4*sin(zb(layer)*m(j))*m(i&
      )*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)&
      *cos(zb(layer)*m(i)) + a_slope*b_slope*zb(layer)**2*m(j)**5*cos(&
      zb(layer)*m(j))*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)&
      **2 - m(j)**6)*sin(zb(layer)*m(i)) - 2*a_slope*b_slope*zt(layer)*&
      sin(zb(layer)*m(j))*m(i)**4*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*&
      m(j)**4*m(i)**2 - m(j)**6)*sin(zb(layer)*m(i)) - 4*a_slope*&
      b_slope*zt(layer)*m(j)*cos(zb(layer)*m(j))*m(i)**3*1.0/(m(i)**6 -&
      3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(zb(layer)*m(&
      i)) + 4*a_slope*b_slope*zt(layer)*m(j)**3*cos(zb(layer)*m(j))*m(i&
      )*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)&
      *cos(zb(layer)*m(i)) + 2*a_slope*b_slope*zt(layer)*m(j)**4*sin(zb&
      (layer)*m(j))*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**&
      2 - m(j)**6)*sin(zb(layer)*m(i)) - a_slope*b_slope*zt(layer)**2*&
      sin(zb(layer)*m(j))*m(i)**5*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*&
      m(j)**4*m(i)**2 - m(j)**6)*cos(zb(layer)*m(i)) + a_slope*b_slope*&
      zt(layer)**2*m(j)*cos(zb(layer)*m(j))*m(i)**4*1.0/(m(i)**6 - 3*m(&
      j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*sin(zb(layer)*m(i))&
      + 2*a_slope*b_slope*zt(layer)**2*m(j)**2*sin(zb(layer)*m(j))*m(i)&
      **3*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**&
      6)*cos(zb(layer)*m(i)) - 2*a_slope*b_slope*zt(layer)**2*m(j)**3*&
      cos(zb(layer)*m(j))*m(i)**2*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*&
      m(j)**4*m(i)**2 - m(j)**6)*sin(zb(layer)*m(i)) - a_slope*b_slope*&
      zt(layer)**2*m(j)**4*sin(zb(layer)*m(j))*m(i)*1.0/(m(i)**6 - 3*m(&
      j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(zb(layer)*m(i))&
      + a_slope*b_slope*zt(layer)**2*m(j)**5*cos(zb(layer)*m(j))*1.0/(m&
      (i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*sin(zb(&
      layer)*m(i)) + a_slope*bt(layer)*sin(zb(layer)*m(j))*m(i)**4*1.0/&
      (m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*sin(&
      zb(layer)*m(i)) + 2*a_slope*bt(layer)*m(j)*cos(zb(layer)*m(j))*m(&
      i)**3*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)&
      **6)*cos(zb(layer)*m(i)) - 2*a_slope*bt(layer)*m(j)**3*cos(zb(&
      layer)*m(j))*m(i)*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(&
      i)**2 - m(j)**6)*cos(zb(layer)*m(i)) - a_slope*bt(layer)*m(j)**4*&
      sin(zb(layer)*m(j))*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*&
      m(i)**2 - m(j)**6)*sin(zb(layer)*m(i)) - a_slope*zb(layer)*bt(&
      layer)*sin(zb(layer)*m(j))*m(i)**5*1.0/(m(i)**6 - 3*m(j)**2*m(i)&
      **4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(zb(layer)*m(i)) + a_slope*&
      zb(layer)*bt(layer)*m(j)*cos(zb(layer)*m(j))*m(i)**4*1.0/(m(i)**6&
      - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*sin(zb(layer)*&
      m(i)) + 2*a_slope*zb(layer)*bt(layer)*m(j)**2*sin(zb(layer)*m(j))&
      *m(i)**3*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m&
      (j)**6)*cos(zb(layer)*m(i)) - 2*a_slope*zb(layer)*bt(layer)*m(j)&
      **3*cos(zb(layer)*m(j))*m(i)**2*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4&
      + 3*m(j)**4*m(i)**2 - m(j)**6)*sin(zb(layer)*m(i)) - a_slope*zb(&
      layer)*bt(layer)*m(j)**4*sin(zb(layer)*m(j))*m(i)*1.0/(m(i)**6 -&
      3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(zb(layer)*m(&
      i)) + a_slope*zb(layer)*bt(layer)*m(j)**5*cos(zb(layer)*m(j))*1.0&
      /(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*sin(&
      zb(layer)*m(i)) + a_slope*zt(layer)*bt(layer)*sin(zb(layer)*m(j))&
      *m(i)**5*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m&
      (j)**6)*cos(zb(layer)*m(i)) - a_slope*zt(layer)*bt(layer)*m(j)*&
      cos(zb(layer)*m(j))*m(i)**4*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*&
      m(j)**4*m(i)**2 - m(j)**6)*sin(zb(layer)*m(i)) - 2*a_slope*zt(&
      layer)*bt(layer)*m(j)**2*sin(zb(layer)*m(j))*m(i)**3*1.0/(m(i)**6&
      - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(zb(layer)*&
      m(i)) + 2*a_slope*zt(layer)*bt(layer)*m(j)**3*cos(zb(layer)*m(j))&
      *m(i)**2*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m&
      (j)**6)*sin(zb(layer)*m(i)) + a_slope*zt(layer)*bt(layer)*m(j)**4&
      *sin(zb(layer)*m(j))*m(i)*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(&
      j)**4*m(i)**2 - m(j)**6)*cos(zb(layer)*m(i)) - a_slope*zt(layer)*&
      bt(layer)*m(j)**5*cos(zb(layer)*m(j))*1.0/(m(i)**6 - 3*m(j)**2*m(&
      i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*sin(zb(layer)*m(i)) +&
      b_slope*at(layer)*sin(zb(layer)*m(j))*m(i)**4*1.0/(m(i)**6 - 3*m(&
      j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*sin(zb(layer)*m(i))&
      + 2*b_slope*at(layer)*m(j)*cos(zb(layer)*m(j))*m(i)**3*1.0/(m(i)&
      **6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(zb(&
      layer)*m(i)) - 2*b_slope*at(layer)*m(j)**3*cos(zb(layer)*m(j))*m(&
      i)*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6&
      )*cos(zb(layer)*m(i)) - b_slope*at(layer)*m(j)**4*sin(zb(layer)*m&
      (j))*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)&
      **6)*sin(zb(layer)*m(i)) - b_slope*zb(layer)*at(layer)*sin(zb(&
      layer)*m(j))*m(i)**5*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4&
      *m(i)**2 - m(j)**6)*cos(zb(layer)*m(i)) + b_slope*zb(layer)*at(&
      layer)*m(j)*cos(zb(layer)*m(j))*m(i)**4*1.0/(m(i)**6 - 3*m(j)**2*&
      m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*sin(zb(layer)*m(i)) + 2*&
      b_slope*zb(layer)*at(layer)*m(j)**2*sin(zb(layer)*m(j))*m(i)**3*&
      1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*&
      cos(zb(layer)*m(i)) - 2*b_slope*zb(layer)*at(layer)*m(j)**3*cos(&
      zb(layer)*m(j))*m(i)**2*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)&
      **4*m(i)**2 - m(j)**6)*sin(zb(layer)*m(i)) - b_slope*zb(layer)*at&
      (layer)*m(j)**4*sin(zb(layer)*m(j))*m(i)*1.0/(m(i)**6 - 3*m(j)**2&
      *m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(zb(layer)*m(i)) +&
      b_slope*zb(layer)*at(layer)*m(j)**5*cos(zb(layer)*m(j))*1.0/(m(i)&
      **6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*sin(zb(&
      layer)*m(i)) + b_slope*zt(layer)*at(layer)*sin(zb(layer)*m(j))*m(&
      i)**5*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)&
      **6)*cos(zb(layer)*m(i)) - b_slope*zt(layer)*at(layer)*m(j)*cos(&
      zb(layer)*m(j))*m(i)**4*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)&
      **4*m(i)**2 - m(j)**6)*sin(zb(layer)*m(i)) - 2*b_slope*zt(layer)*&
      at(layer)*m(j)**2*sin(zb(layer)*m(j))*m(i)**3*1.0/(m(i)**6 - 3*m(&
      j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(zb(layer)*m(i))&
      + 2*b_slope*zt(layer)*at(layer)*m(j)**3*cos(zb(layer)*m(j))*m(i)&
      **2*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**&
      6)*sin(zb(layer)*m(i)) + b_slope*zt(layer)*at(layer)*m(j)**4*sin(&
      zb(layer)*m(j))*m(i)*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4&
      *m(i)**2 - m(j)**6)*cos(zb(layer)*m(i)) - b_slope*zt(layer)*at(&
      layer)*m(j)**5*cos(zb(layer)*m(j))*1.0/(m(i)**6 - 3*m(j)**2*m(i)&
      **4 + 3*m(j)**4*m(i)**2 - m(j)**6)*sin(zb(layer)*m(i)) - bt(layer&
      )*at(layer)*sin(zb(layer)*m(j))*m(i)**5*1.0/(m(i)**6 - 3*m(j)**2*&
      m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*cos(zb(layer)*m(i)) + bt(&
      layer)*at(layer)*m(j)*cos(zb(layer)*m(j))*m(i)**4*1.0/(m(i)**6 -&
      3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*sin(zb(layer)*m(&
      i)) + 2*bt(layer)*at(layer)*m(j)**2*sin(zb(layer)*m(j))*m(i)**3*&
      1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)**2 - m(j)**6)*&
      cos(zb(layer)*m(i)) - 2*bt(layer)*at(layer)*m(j)**3*cos(zb(layer)&
      *m(j))*m(i)**2*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j)**4*m(i)&
      **2 - m(j)**6)*sin(zb(layer)*m(i)) - bt(layer)*at(layer)*m(j)**4*&
      sin(zb(layer)*m(j))*m(i)*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3*m(j&
      )**4*m(i)**2 - m(j)**6)*cos(zb(layer)*m(i)) + bt(layer)*at(layer)&
      *m(j)**5*cos(zb(layer)*m(j))*1.0/(m(i)**6 - 3*m(j)**2*m(i)**4 + 3&
      *m(j)**4*m(i)**2 - m(j)**6)*sin(zb(layer)*m(i))))
            END DO
          END DO
        END DO

        DO j = 0, neig -2
          DO i = j + 1, neig-1
            a(j,i) = a(i, j)
          END DO
        END DO

      END SUBROUTINE    

      
      

      SUBROUTINE eload_sinlinear(loadtim, loadmag, omega, phase, &
                                 eigs, tvals, dT, a, neig, nload, nt)
        USE types
        IMPLICIT NONE

        INTEGER, intent(in) :: neig
        INTEGER, intent(in) :: nload
        INTEGER, intent(in) :: nt
        REAL(DP), intent(in), dimension(0:nload-1) :: loadtim
        REAL(DP), intent(in), dimension(0:nload-1) :: loadmag
        REAL(DP), intent(in), dimension(0:neig-1) :: eigs
        REAL(DP), intent(in), dimension(0:nt-1) :: tvals
        REAL(DP), intent(in) :: dT
        REAL(DP), intent(in) :: omega
        REAL(DP), intent(in) :: phase
        REAL(DP), intent(out), dimension(0:nt-1, 0:neig-1) :: a
        INTEGER :: i , j, k
        REAL(DP):: EPSILON
        a=0.0D0
        EPSILON = 0.0000005D0
        DO i = 0, nt-1
          DO k = 0, nload-2

            IF (tvals(i) < loadtim(k)) EXIT !t is before load step

            IF (tvals(i) >= loadtim(k + 1)) THEN
              !t is after the load step
              IF(ABS(loadtim(k) - loadtim(k + 1)) <= &
                (ABS(loadtim(k) + loadtim(k + 1))*EPSILON)) THEN
                !step load
                CONTINUE
              ELSEIF(ABS(loadmag(k) - loadmag(k + 1)) <= &
                    (ABS(loadmag(k) + loadmag(k + 1))*EPSILON)) THEN
                !constant load
                DO j=0, neig-1
      a(i, j) = a(i, j) + (1.0/eigs(j)*(1.0/(1 + omega**2*eigs(j)**(-2)/&
      dT**2)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*sin(-omega*(&
      -loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase) - omega*1.0&
      /eigs(j)*1.0/(1 + omega**2*eigs(j)**(-2)/dT**2)*cos(-omega*(&
      -loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*&
      eigs(j)*(-loadtim(k + 1) + tvals(i)))/dT)*loadmag(k)/dT - 1.0/&
      eigs(j)*(1.0/(1 + omega**2*eigs(j)**(-2)/dT**2)*exp(-dT*eigs(j)*(&
      -loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals(i)) +&
      omega*tvals(i) + phase) - omega*1.0/eigs(j)*1.0/(1 + omega**2*&
      eigs(j)**(-2)/dT**2)*cos(-omega*(-loadtim(k) + tvals(i)) + omega*&
      tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))/dT)*&
      loadmag(k)/dT)
                END DO
              ELSE
                !ramp load
                DO j=0, neig-1
      a(i, j) = a(i, j) + (1.0/eigs(j)*(dT*eigs(j)*(-loadtim(k + 1) +&
      tvals(i))*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*sin(&
      -omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase)*1.0&
      /(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2&
      *1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k&
      )/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*&
      eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1) - dT*eigs(j)*(&
      -loadtim(k + 1) + tvals(i))*exp(-dT*eigs(j)*(-loadtim(k + 1) +&
      tvals(i)))*sin(-omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(&
      i) + phase)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(&
      k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/&
      eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT&
      **3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k) + dT*&
      eigs(j)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*sin(-omega*&
      (-loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase)*loadtim(k)&
      *1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*&
      omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1) - dT*eigs&
      (j)*loadtim(k + 1)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*&
      sin(-omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase)&
      *1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*&
      omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k) - dT*tvals(i)&
      *eigs(j)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*sin(-omega&
      *(-loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase)*1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*loadmag(k + 1) + dT*tvals(i)*eigs(j)*exp&
      (-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*sin(-omega*(-loadtim(k&
      + 1) + tvals(i)) + omega*tvals(i) + phase)*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*loadmag(k) - omega*(-loadtim(k + 1) + tvals(i))&
      *cos(-omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase&
      )*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*loadmag(k + 1) + omega*(-loadtim(k + 1) + tvals&
      (i))*cos(-omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(i) +&
      phase)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*loadmag(k) - omega*cos(-omega*(-loadtim(&
      k + 1) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(&
      -loadtim(k + 1) + tvals(i)))*loadtim(k)*1.0/(-dT*eigs(j)*loadtim(&
      k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k&
      + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)&
      **(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/&
      dT**3)*loadmag(k + 1) + omega*loadtim(k + 1)*cos(-omega*(-loadtim&
      (k + 1) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(&
      -loadtim(k + 1) + tvals(i)))*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT&
      *eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT +&
      2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      loadmag(k) + omega*tvals(i)*cos(-omega*(-loadtim(k + 1) + tvals(i&
      )) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k + 1) +&
      tvals(i)))*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k&
      ) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/&
      eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT&
      **3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1) -&
      omega*tvals(i)*cos(-omega*(-loadtim(k + 1) + tvals(i)) + omega*&
      tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*&
      1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega&
      **2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k) + exp(-dT*&
      eigs(j)*(-loadtim(k + 1) + tvals(i)))*sin(-omega*(-loadtim(k + 1&
      ) + tvals(i)) + omega*tvals(i) + phase)*1.0/(-dT*eigs(j)*loadtim(&
      k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k&
      + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)&
      **(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/&
      dT**3)*loadmag(k + 1) - exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(&
      i)))*sin(-omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(i) +&
      phase)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) -&
      2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)&
      *loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k) + omega**2*&
      1.0/eigs(j)*(-loadtim(k + 1) + tvals(i))*exp(-dT*eigs(j)*(&
      -loadtim(k + 1) + tvals(i)))*sin(-omega*(-loadtim(k + 1) + tvals(&
      i)) + omega*tvals(i) + phase)*1.0/(-dT*eigs(j)*loadtim(k + 1) +&
      dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT&
      + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      loadmag(k + 1)/dT - omega**2*1.0/eigs(j)*(-loadtim(k + 1) + tvals&
      (i))*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*sin(-omega*(&
      -loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase)*1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*loadmag(k)/dT + omega**2*1.0/eigs(j)*exp&
      (-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*sin(-omega*(-loadtim(k&
      + 1) + tvals(i)) + omega*tvals(i) + phase)*loadtim(k)*1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*loadmag(k + 1)/dT - omega**2*1.0/eigs(j)&
      *loadtim(k + 1)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*sin&
      (-omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase)*&
      1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega&
      **2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k)/dT - omega**2&
      *tvals(i)*1.0/eigs(j)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i&
      )))*sin(-omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(i) +&
      phase)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) -&
      2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)&
      *loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1)/dT +&
      omega**2*tvals(i)*1.0/eigs(j)*exp(-dT*eigs(j)*(-loadtim(k + 1) +&
      tvals(i)))*sin(-omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(&
      i) + phase)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(&
      k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/&
      eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT&
      **3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k)/dT - 2*&
      omega*1.0/eigs(j)*cos(-omega*(-loadtim(k + 1) + tvals(i)) + omega&
      *tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*&
      1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega&
      **2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1)/dT + 2*&
      omega*1.0/eigs(j)*cos(-omega*(-loadtim(k + 1) + tvals(i)) + omega&
      *tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*&
      1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega&
      **2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k)/dT - omega**3&
      *eigs(j)**(-2)*(-loadtim(k + 1) + tvals(i))*cos(-omega*(-loadtim(&
      k + 1) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(&
      -loadtim(k + 1) + tvals(i)))*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT&
      *eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT +&
      2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      loadmag(k + 1)/dT**2 + omega**3*eigs(j)**(-2)*(-loadtim(k + 1) +&
      tvals(i))*cos(-omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(i&
      ) + phase)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*1.0/(-dT&
      *eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*loadmag(k)/dT**2 - omega**3*eigs(j)**(-2&
      )*cos(-omega*(-loadtim(k + 1) + tvals(i)) + omega*tvals(i) +&
      phase)*exp(-dT*eigs(j)*(-loadtim(k + 1) + tvals(i)))*loadtim(k)*&
      1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega&
      **2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1)/dT**2 +&
      omega**3*eigs(j)**(-2)*loadtim(k + 1)*cos(-omega*(-loadtim(k + 1&
      ) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim&
      (k + 1) + tvals(i)))*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)&
      *loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega&
      **2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k&
      + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k)/&
      dT**2 + omega**3*tvals(i)*eigs(j)**(-2)*cos(-omega*(-loadtim(k +&
      1) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(&
      -loadtim(k + 1) + tvals(i)))*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT&
      *eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT +&
      2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      loadmag(k + 1)/dT**2 - omega**3*tvals(i)*eigs(j)**(-2)*cos(-omega&
      *(-loadtim(k + 1) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*&
      eigs(j)*(-loadtim(k + 1) + tvals(i)))*1.0/(-dT*eigs(j)*loadtim(k&
      + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k +&
      1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**&
      (-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT&
      **3)*loadmag(k)/dT**2 - omega**2*eigs(j)**(-2)*exp(-dT*eigs(j)*(&
      -loadtim(k + 1) + tvals(i)))*sin(-omega*(-loadtim(k + 1) + tvals(&
      i)) + omega*tvals(i) + phase)*1.0/(-dT*eigs(j)*loadtim(k + 1) +&
      dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT&
      + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      loadmag(k + 1)/dT**2 + omega**2*eigs(j)**(-2)*exp(-dT*eigs(j)*(&
      -loadtim(k + 1) + tvals(i)))*sin(-omega*(-loadtim(k + 1) + tvals(&
      i)) + omega*tvals(i) + phase)*1.0/(-dT*eigs(j)*loadtim(k + 1) +&
      dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT&
      + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      loadmag(k)/dT**2)/dT - 1.0/eigs(j)*(dT*eigs(j)*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*(-loadtim(k) + tvals(i))*exp(-dT*eigs(j)*(&
      -loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals(i)) +&
      omega*tvals(i) + phase)*loadmag(k + 1) - dT*eigs(j)*1.0/(-dT*eigs&
      (j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(&
      j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT -&
      omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(&
      -3)*loadtim(k)/dT**3)*(-loadtim(k) + tvals(i))*exp(-dT*eigs(j)*(&
      -loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals(i)) +&
      omega*tvals(i) + phase)*loadmag(k) - dT*eigs(j)*loadtim(k + 1)*&
      1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega&
      **2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(j)*(&
      -loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals(i)) +&
      omega*tvals(i) + phase)*loadmag(k) + dT*eigs(j)*loadtim(k)*1.0/(&
      -dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*&
      1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)&
      /dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs&
      (j)**(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(&
      i)))*sin(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase&
      )*loadmag(k + 1) - dT*tvals(i)*eigs(j)*1.0/(-dT*eigs(j)*loadtim(k&
      + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k +&
      1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**&
      (-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT&
      **3)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(&
      -loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*loadmag(k + 1)&
      + dT*tvals(i)*eigs(j)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j&
      )*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega&
      **2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k&
      + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*exp(-dT*&
      eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals&
      (i)) + omega*tvals(i) + phase)*loadmag(k) - omega*1.0/(-dT*eigs(j&
      )*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)&
      *loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*(-loadtim(k) + tvals(i))*cos(-omega*(-loadtim(k&
      ) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim&
      (k) + tvals(i)))*loadmag(k + 1) + omega*1.0/(-dT*eigs(j)*loadtim(&
      k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k&
      + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)&
      **(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/&
      dT**3)*(-loadtim(k) + tvals(i))*cos(-omega*(-loadtim(k) + tvals(i&
      )) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals&
      (i)))*loadmag(k) + omega*loadtim(k + 1)*1.0/(-dT*eigs(j)*loadtim(&
      k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k&
      + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)&
      **(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/&
      dT**3)*cos(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) +&
      phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k) -&
      omega*loadtim(k)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*cos(-omega*(&
      -loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)&
      *(-loadtim(k) + tvals(i)))*loadmag(k + 1) + omega*tvals(i)*1.0/(&
      -dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*&
      1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)&
      /dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs&
      (j)**(-3)*loadtim(k)/dT**3)*cos(-omega*(-loadtim(k) + tvals(i)) +&
      omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))&
      *loadmag(k + 1) - omega*tvals(i)*1.0/(-dT*eigs(j)*loadtim(k + 1)&
      + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/&
      dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3&
      )*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)&
      *cos(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*&
      exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k) + 1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i&
      )))*sin(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)&
      *loadmag(k + 1) - 1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(&
      j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals(i))&
      + omega*tvals(i) + phase)*loadmag(k) + omega**2*1.0/eigs(j)*1.0/(&
      -dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*&
      1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)&
      /dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs&
      (j)**(-3)*loadtim(k)/dT**3)*(-loadtim(k) + tvals(i))*exp(-dT*eigs&
      (j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals(i&
      )) + omega*tvals(i) + phase)*loadmag(k + 1)/dT - omega**2*1.0/&
      eigs(j)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) -&
      2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)&
      *loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*(-loadtim(k) + tvals(i))&
      *exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k&
      ) + tvals(i)) + omega*tvals(i) + phase)*loadmag(k)/dT - omega**2*&
      1.0/eigs(j)*loadtim(k + 1)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*&
      eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2&
      *omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k&
      ) + tvals(i)) + omega*tvals(i) + phase)*loadmag(k)/dT + omega**2*&
      1.0/eigs(j)*loadtim(k)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(&
      j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*&
      omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k&
      ) + tvals(i)) + omega*tvals(i) + phase)*loadmag(k + 1)/dT - omega&
      **2*tvals(i)*1.0/eigs(j)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*&
      eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2&
      *omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k&
      ) + tvals(i)) + omega*tvals(i) + phase)*loadmag(k + 1)/dT + omega&
      **2*tvals(i)*1.0/eigs(j)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*&
      eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2&
      *omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k&
      ) + tvals(i)) + omega*tvals(i) + phase)*loadmag(k)/dT - 2*omega*&
      1.0/eigs(j)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(&
      k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/&
      eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT&
      **3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*cos(-omega*(&
      -loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)&
      *(-loadtim(k) + tvals(i)))*loadmag(k + 1)/dT + 2*omega*1.0/eigs(j&
      )*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*&
      omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*cos(-omega*(-loadtim(k)&
      + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k&
      ) + tvals(i)))*loadmag(k)/dT - omega**3*eigs(j)**(-2)*1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*(-loadtim(k) + tvals(i))*cos(-omega*(&
      -loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)&
      *(-loadtim(k) + tvals(i)))*loadmag(k + 1)/dT**2 + omega**3*eigs(j&
      )**(-2)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) -&
      2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)&
      *loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*(-loadtim(k) + tvals(i))&
      *cos(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*&
      exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k)/dT**2 +&
      omega**3*eigs(j)**(-2)*loadtim(k + 1)*1.0/(-dT*eigs(j)*loadtim(k&
      + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k +&
      1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**&
      (-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT&
      **3)*cos(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase&
      )*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k)/dT**2 -&
      omega**3*eigs(j)**(-2)*loadtim(k)*1.0/(-dT*eigs(j)*loadtim(k + 1&
      ) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)&
      /dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(&
      -3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**&
      3)*cos(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*&
      exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k + 1)/dT**2 +&
      omega**3*tvals(i)*eigs(j)**(-2)*1.0/(-dT*eigs(j)*loadtim(k + 1) +&
      dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT&
      + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      cos(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*exp&
      (-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k + 1)/dT**2 -&
      omega**3*tvals(i)*eigs(j)**(-2)*1.0/(-dT*eigs(j)*loadtim(k + 1) +&
      dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT&
      + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      cos(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*exp&
      (-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k)/dT**2 - omega**&
      2*eigs(j)**(-2)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(&
      j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals(i))&
      + omega*tvals(i) + phase)*loadmag(k + 1)/dT**2 + omega**2*eigs(j)&
      **(-2)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) -&
      2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)&
      *loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(j)*(&
      -loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals(i)) +&
      omega*tvals(i) + phase)*loadmag(k)/dT**2)/dT)
                END DO
              END IF
            ELSE
              !t is in the load step
              IF(ABS(loadmag(k) - loadmag(k + 1)) <= &
                    (ABS(loadmag(k) + loadmag(k + 1))*EPSILON)) THEN
                !constant load
                DO j=0, neig-1
      a(i, j) = a(i, j) + (1.0/eigs(j)*(sin(omega*tvals(i) + phase)*1.0/&
      (1 + omega**2*eigs(j)**(-2)/dT**2) - omega*cos(omega*tvals(i) +&
      phase)*1.0/eigs(j)*1.0/(1 + omega**2*eigs(j)**(-2)/dT**2)/dT)*&
      loadmag(k)/dT - 1.0/eigs(j)*(1.0/(1 + omega**2*eigs(j)**(-2)/dT**&
      2)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim&
      (k) + tvals(i)) + omega*tvals(i) + phase) - omega*1.0/eigs(j)*1.0&
      /(1 + omega**2*eigs(j)**(-2)/dT**2)*cos(-omega*(-loadtim(k) +&
      tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k)&
      + tvals(i)))/dT)*loadmag(k)/dT)
                END DO
              ELSE
                !ramp load
                DO j=0, neig-1
      a(i, j) = a(i, j) + (1.0/eigs(j)*(-dT*sin(omega*tvals(i) + phase)*&
      eigs(j)*loadtim(k + 1)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(&
      j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*&
      omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      loadmag(k) + dT*sin(omega*tvals(i) + phase)*eigs(j)*loadtim(k)*&
      1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega&
      **2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1) - dT*&
      tvals(i)*sin(omega*tvals(i) + phase)*eigs(j)*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*loadmag(k + 1) + dT*tvals(i)*sin(omega*tvals(i&
      ) + phase)*eigs(j)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k) +&
      omega*cos(omega*tvals(i) + phase)*loadtim(k + 1)*1.0/(-dT*eigs(j)&
      *loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*loadmag(k) - omega*cos(omega*tvals(i) + phase)*&
      loadtim(k)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k&
      ) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/&
      eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT&
      **3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1) +&
      omega*tvals(i)*cos(omega*tvals(i) + phase)*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*loadmag(k + 1) - omega*tvals(i)*cos(omega*tvals&
      (i) + phase)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim&
      (k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/&
      eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT&
      **3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k) + sin(&
      omega*tvals(i) + phase)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs&
      (j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*&
      omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      loadmag(k + 1) - sin(omega*tvals(i) + phase)*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*loadmag(k) - omega**2*sin(omega*tvals(i) +&
      phase)*1.0/eigs(j)*loadtim(k + 1)*1.0/(-dT*eigs(j)*loadtim(k + 1&
      ) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)&
      /dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(&
      -3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**&
      3)*loadmag(k)/dT + omega**2*sin(omega*tvals(i) + phase)*1.0/eigs(&
      j)*loadtim(k)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1&
      )/dT - omega**2*tvals(i)*sin(omega*tvals(i) + phase)*1.0/eigs(j)*&
      1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega&
      **2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1)/dT +&
      omega**2*tvals(i)*sin(omega*tvals(i) + phase)*1.0/eigs(j)*1.0/(&
      -dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*&
      1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)&
      /dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs&
      (j)**(-3)*loadtim(k)/dT**3)*loadmag(k)/dT - 2*omega*cos(omega*&
      tvals(i) + phase)*1.0/eigs(j)*1.0/(-dT*eigs(j)*loadtim(k + 1) +&
      dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT&
      + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      loadmag(k + 1)/dT + 2*omega*cos(omega*tvals(i) + phase)*1.0/eigs(&
      j)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*&
      omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k)/dT + omega**3&
      *cos(omega*tvals(i) + phase)*eigs(j)**(-2)*loadtim(k + 1)*1.0/(&
      -dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*&
      1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)&
      /dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs&
      (j)**(-3)*loadtim(k)/dT**3)*loadmag(k)/dT**2 - omega**3*cos(omega&
      *tvals(i) + phase)*eigs(j)**(-2)*loadtim(k)*1.0/(-dT*eigs(j)*&
      loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*&
      loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*loadmag(k + 1)/dT**2 + omega**3*tvals(i)*cos(&
      omega*tvals(i) + phase)*eigs(j)**(-2)*1.0/(-dT*eigs(j)*loadtim(k&
      + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k +&
      1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**&
      (-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT&
      **3)*loadmag(k + 1)/dT**2 - omega**3*tvals(i)*cos(omega*tvals(i)&
      + phase)*eigs(j)**(-2)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(&
      j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*&
      omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      loadmag(k)/dT**2 - omega**2*sin(omega*tvals(i) + phase)*eigs(j)**&
      (-2)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*&
      omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*loadmag(k + 1)/dT**2 +&
      omega**2*sin(omega*tvals(i) + phase)*eigs(j)**(-2)*1.0/(-dT*eigs(&
      j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j&
      )*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT -&
      omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(&
      -3)*loadtim(k)/dT**3)*loadmag(k)/dT**2)/dT - 1.0/eigs(j)*(dT*eigs&
      (j)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*&
      omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*(-loadtim(k) + tvals(i))&
      *exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k&
      ) + tvals(i)) + omega*tvals(i) + phase)*loadmag(k + 1) - dT*eigs(&
      j)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*&
      omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*(-loadtim(k) + tvals(i))&
      *exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k&
      ) + tvals(i)) + omega*tvals(i) + phase)*loadmag(k) - dT*eigs(j)*&
      loadtim(k + 1)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(&
      j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals(i))&
      + omega*tvals(i) + phase)*loadmag(k) + dT*eigs(j)*loadtim(k)*1.0/&
      (-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*&
      1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)&
      /dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs&
      (j)**(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(&
      i)))*sin(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase&
      )*loadmag(k + 1) - dT*tvals(i)*eigs(j)*1.0/(-dT*eigs(j)*loadtim(k&
      + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k +&
      1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**&
      (-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT&
      **3)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(&
      -loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*loadmag(k + 1)&
      + dT*tvals(i)*eigs(j)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j&
      )*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega&
      **2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k&
      + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*exp(-dT*&
      eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals&
      (i)) + omega*tvals(i) + phase)*loadmag(k) - omega*1.0/(-dT*eigs(j&
      )*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)&
      *loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega&
      **4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*&
      loadtim(k)/dT**3)*(-loadtim(k) + tvals(i))*cos(-omega*(-loadtim(k&
      ) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim&
      (k) + tvals(i)))*loadmag(k + 1) + omega*1.0/(-dT*eigs(j)*loadtim(&
      k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k&
      + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)&
      **(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/&
      dT**3)*(-loadtim(k) + tvals(i))*cos(-omega*(-loadtim(k) + tvals(i&
      )) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals&
      (i)))*loadmag(k) + omega*loadtim(k + 1)*1.0/(-dT*eigs(j)*loadtim(&
      k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k&
      + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)&
      **(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/&
      dT**3)*cos(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) +&
      phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k) -&
      omega*loadtim(k)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*cos(-omega*(&
      -loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)&
      *(-loadtim(k) + tvals(i)))*loadmag(k + 1) + omega*tvals(i)*1.0/(&
      -dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*&
      1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)&
      /dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs&
      (j)**(-3)*loadtim(k)/dT**3)*cos(-omega*(-loadtim(k) + tvals(i)) +&
      omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))&
      *loadmag(k + 1) - omega*tvals(i)*1.0/(-dT*eigs(j)*loadtim(k + 1)&
      + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/&
      dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3&
      )*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)&
      *cos(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*&
      exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k) + 1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i&
      )))*sin(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)&
      *loadmag(k + 1) - 1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(&
      j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals(i))&
      + omega*tvals(i) + phase)*loadmag(k) + omega**2*1.0/eigs(j)*1.0/(&
      -dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*&
      1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)&
      /dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs&
      (j)**(-3)*loadtim(k)/dT**3)*(-loadtim(k) + tvals(i))*exp(-dT*eigs&
      (j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals(i&
      )) + omega*tvals(i) + phase)*loadmag(k + 1)/dT - omega**2*1.0/&
      eigs(j)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) -&
      2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)&
      *loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*(-loadtim(k) + tvals(i))&
      *exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k&
      ) + tvals(i)) + omega*tvals(i) + phase)*loadmag(k)/dT - omega**2*&
      1.0/eigs(j)*loadtim(k + 1)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*&
      eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2&
      *omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k&
      ) + tvals(i)) + omega*tvals(i) + phase)*loadmag(k)/dT + omega**2*&
      1.0/eigs(j)*loadtim(k)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(&
      j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*&
      omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k&
      ) + tvals(i)) + omega*tvals(i) + phase)*loadmag(k + 1)/dT - omega&
      **2*tvals(i)*1.0/eigs(j)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*&
      eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2&
      *omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k&
      ) + tvals(i)) + omega*tvals(i) + phase)*loadmag(k + 1)/dT + omega&
      **2*tvals(i)*1.0/eigs(j)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*&
      eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2&
      *omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k&
      ) + tvals(i)) + omega*tvals(i) + phase)*loadmag(k)/dT - 2*omega*&
      1.0/eigs(j)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(&
      k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/&
      eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT&
      **3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*cos(-omega*(&
      -loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)&
      *(-loadtim(k) + tvals(i)))*loadmag(k + 1)/dT + 2*omega*1.0/eigs(j&
      )*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*&
      omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*&
      loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*cos(-omega*(-loadtim(k)&
      + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)*(-loadtim(k&
      ) + tvals(i)))*loadmag(k)/dT - omega**3*eigs(j)**(-2)*1.0/(-dT*&
      eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/&
      eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT&
      - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)&
      **(-3)*loadtim(k)/dT**3)*(-loadtim(k) + tvals(i))*cos(-omega*(&
      -loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*exp(-dT*eigs(j)&
      *(-loadtim(k) + tvals(i)))*loadmag(k + 1)/dT**2 + omega**3*eigs(j&
      )**(-2)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) -&
      2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)&
      *loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*(-loadtim(k) + tvals(i))&
      *cos(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*&
      exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k)/dT**2 +&
      omega**3*eigs(j)**(-2)*loadtim(k + 1)*1.0/(-dT*eigs(j)*loadtim(k&
      + 1) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k +&
      1)/dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**&
      (-3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT&
      **3)*cos(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase&
      )*exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k)/dT**2 -&
      omega**3*eigs(j)**(-2)*loadtim(k)*1.0/(-dT*eigs(j)*loadtim(k + 1&
      ) + dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)&
      /dT + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(&
      -3)*loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**&
      3)*cos(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*&
      exp(-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k + 1)/dT**2 +&
      omega**3*tvals(i)*eigs(j)**(-2)*1.0/(-dT*eigs(j)*loadtim(k + 1) +&
      dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT&
      + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      cos(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*exp&
      (-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k + 1)/dT**2 -&
      omega**3*tvals(i)*eigs(j)**(-2)*1.0/(-dT*eigs(j)*loadtim(k + 1) +&
      dT*eigs(j)*loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT&
      + 2*omega**2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*&
      loadtim(k + 1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*&
      cos(-omega*(-loadtim(k) + tvals(i)) + omega*tvals(i) + phase)*exp&
      (-dT*eigs(j)*(-loadtim(k) + tvals(i)))*loadmag(k)/dT**2 - omega**&
      2*eigs(j)**(-2)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*&
      loadtim(k) - 2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**&
      2*1.0/eigs(j)*loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k +&
      1)/dT**3 + omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(&
      j)*(-loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals(i))&
      + omega*tvals(i) + phase)*loadmag(k + 1)/dT**2 + omega**2*eigs(j)&
      **(-2)*1.0/(-dT*eigs(j)*loadtim(k + 1) + dT*eigs(j)*loadtim(k) -&
      2*omega**2*1.0/eigs(j)*loadtim(k + 1)/dT + 2*omega**2*1.0/eigs(j)&
      *loadtim(k)/dT - omega**4*eigs(j)**(-3)*loadtim(k + 1)/dT**3 +&
      omega**4*eigs(j)**(-3)*loadtim(k)/dT**3)*exp(-dT*eigs(j)*(&
      -loadtim(k) + tvals(i)))*sin(-omega*(-loadtim(k) + tvals(i)) +&
      omega*tvals(i) + phase)*loadmag(k)/dT**2)/dT)
                END DO
              END IF
            END IF
          END DO
        END DO

      END SUBROUTINE



        

    

