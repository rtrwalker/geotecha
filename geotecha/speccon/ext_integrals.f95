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
        a=0.0D0
        DO layer = 0, nlayers-1
          DO j = 0, neig-1
              i=j
      a(i, i) = a(i, i) + (-1.0/(4*zb(layer)*m(i)**2 - 4*zt(layer)*m(i)&
      **2)*ab(layer)*cos(m(i)*zb(layer))**2 + 1.0/(4*zb(layer)*m(i)**2&
      - 4*zt(layer)*m(i)**2)*ab(layer)*cos(m(i)*zt(layer))**2 + 1.0/(4*&
      zb(layer)*m(i)**2 - 4*zt(layer)*m(i)**2)*at(layer)*cos(m(i)*zb(&
      layer))**2 - 1.0/(4*zb(layer)*m(i)**2 - 4*zt(layer)*m(i)**2)*at(&
      layer)*cos(m(i)*zt(layer))**2 - 2*m(i)*1.0/(4*zb(layer)*m(i)**2 -&
      4*zt(layer)*m(i)**2)*ab(layer)*zb(layer)*cos(m(i)*zb(layer))*sin(&
      m(i)*zb(layer)) + 2*m(i)*1.0/(4*zb(layer)*m(i)**2 - 4*zt(layer)*m&
      (i)**2)*ab(layer)*zt(layer)*cos(m(i)*zt(layer))*sin(m(i)*zt(layer&
      )) + 2*m(i)*1.0/(4*zb(layer)*m(i)**2 - 4*zt(layer)*m(i)**2)*at(&
      layer)*zb(layer)*cos(m(i)*zb(layer))*sin(m(i)*zb(layer)) - 2*m(i)&
      *1.0/(4*zb(layer)*m(i)**2 - 4*zt(layer)*m(i)**2)*at(layer)*zt(&
      layer)*cos(m(i)*zt(layer))*sin(m(i)*zt(layer)) + m(i)**2*1.0/(4*&
      zb(layer)*m(i)**2 - 4*zt(layer)*m(i)**2)*ab(layer)*zb(layer)**2*&
      sin(m(i)*zb(layer))**2 + m(i)**2*1.0/(4*zb(layer)*m(i)**2 - 4*zt(&
      layer)*m(i)**2)*ab(layer)*zb(layer)**2*cos(m(i)*zb(layer))**2 - m&
      (i)**2*1.0/(4*zb(layer)*m(i)**2 - 4*zt(layer)*m(i)**2)*ab(layer)*&
      zt(layer)**2*sin(m(i)*zt(layer))**2 - m(i)**2*1.0/(4*zb(layer)*m(&
      i)**2 - 4*zt(layer)*m(i)**2)*ab(layer)*zt(layer)**2*cos(m(i)*zt(&
      layer))**2 - m(i)**2*1.0/(4*zb(layer)*m(i)**2 - 4*zt(layer)*m(i)&
      **2)*at(layer)*zb(layer)**2*sin(m(i)*zb(layer))**2 - m(i)**2*1.0/&
      (4*zb(layer)*m(i)**2 - 4*zt(layer)*m(i)**2)*at(layer)*zb(layer)**&
      2*cos(m(i)*zb(layer))**2 + m(i)**2*1.0/(4*zb(layer)*m(i)**2 - 4*&
      zt(layer)*m(i)**2)*at(layer)*zt(layer)**2*sin(m(i)*zt(layer))**2&
      + m(i)**2*1.0/(4*zb(layer)*m(i)**2 - 4*zt(layer)*m(i)**2)*at(&
      layer)*zt(layer)**2*cos(m(i)*zt(layer))**2 - 2*zb(layer)*m(i)*1.0&
      /(4*zb(layer)*m(i)**2 - 4*zt(layer)*m(i)**2)*at(layer)*cos(m(i)*&
      zb(layer))*sin(m(i)*zb(layer)) + 2*zb(layer)*m(i)*1.0/(4*zb(layer&
      )*m(i)**2 - 4*zt(layer)*m(i)**2)*at(layer)*cos(m(i)*zt(layer))*&
      sin(m(i)*zt(layer)) + 2*zb(layer)*m(i)**2*1.0/(4*zb(layer)*m(i)**&
      2 - 4*zt(layer)*m(i)**2)*at(layer)*zb(layer)*sin(m(i)*zb(layer))&
      **2 + 2*zb(layer)*m(i)**2*1.0/(4*zb(layer)*m(i)**2 - 4*zt(layer)*&
      m(i)**2)*at(layer)*zb(layer)*cos(m(i)*zb(layer))**2 - 2*zb(layer)&
      *m(i)**2*1.0/(4*zb(layer)*m(i)**2 - 4*zt(layer)*m(i)**2)*at(layer&
      )*zt(layer)*sin(m(i)*zt(layer))**2 - 2*zb(layer)*m(i)**2*1.0/(4*&
      zb(layer)*m(i)**2 - 4*zt(layer)*m(i)**2)*at(layer)*zt(layer)*cos(&
      m(i)*zt(layer))**2 + 2*zt(layer)*m(i)*1.0/(4*zb(layer)*m(i)**2 -&
      4*zt(layer)*m(i)**2)*ab(layer)*cos(m(i)*zb(layer))*sin(m(i)*zb(&
      layer)) - 2*zt(layer)*m(i)*1.0/(4*zb(layer)*m(i)**2 - 4*zt(layer)&
      *m(i)**2)*ab(layer)*cos(m(i)*zt(layer))*sin(m(i)*zt(layer)) - 2*&
      zt(layer)*m(i)**2*1.0/(4*zb(layer)*m(i)**2 - 4*zt(layer)*m(i)**2)&
      *ab(layer)*zb(layer)*sin(m(i)*zb(layer))**2 - 2*zt(layer)*m(i)**2&
      *1.0/(4*zb(layer)*m(i)**2 - 4*zt(layer)*m(i)**2)*ab(layer)*zb(&
      layer)*cos(m(i)*zb(layer))**2 + 2*zt(layer)*m(i)**2*1.0/(4*zb(&
      layer)*m(i)**2 - 4*zt(layer)*m(i)**2)*ab(layer)*zt(layer)*sin(m(i&
      )*zt(layer))**2 + 2*zt(layer)*m(i)**2*1.0/(4*zb(layer)*m(i)**2 -&
      4*zt(layer)*m(i)**2)*ab(layer)*zt(layer)*cos(m(i)*zt(layer))**2)
            DO i = j+1, neig-1
      a(i, j) = a(i, j) + (m(i)**2*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*&
      m(j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(&
      layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)*ab(layer)*sin(m(i)*zb&
      (layer))*sin(m(j)*zb(layer)) - m(i)**2*1.0/(zb(layer)*m(i)**4 - 2&
      *zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**&
      4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)*ab(layer)*&
      sin(m(i)*zt(layer))*sin(m(j)*zt(layer)) - m(i)**2*1.0/(zb(layer)*&
      m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(&
      layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)&
      *at(layer)*sin(m(i)*zb(layer))*sin(m(j)*zb(layer)) + m(i)**2*1.0/&
      (zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)&
      **4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)&
      *m(j)**4)*at(layer)*sin(m(i)*zt(layer))*sin(m(j)*zt(layer)) - m(i&
      )**3*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(&
      layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2&
      - zt(layer)*m(j)**4)*ab(layer)*zb(layer)*cos(m(i)*zb(layer))*sin(&
      m(j)*zb(layer)) + m(i)**3*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(&
      j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(&
      layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)*ab(layer)*zt(layer)*&
      cos(m(i)*zt(layer))*sin(m(j)*zt(layer)) + m(i)**3*1.0/(zb(layer)*&
      m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(&
      layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)&
      *at(layer)*zb(layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) - m(&
      i)**3*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(&
      layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2&
      - zt(layer)*m(j)**4)*at(layer)*zt(layer)*cos(m(i)*zt(layer))*sin(&
      m(j)*zt(layer)) + 2*m(j)*m(i)*1.0/(zb(layer)*m(i)**4 - 2*zb(layer&
      )*m(j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(&
      layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)*ab(layer)*cos(m(i)*zb&
      (layer))*cos(m(j)*zb(layer)) - 2*m(j)*m(i)*1.0/(zb(layer)*m(i)**4&
      - 2*zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i&
      )**4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)*ab(layer)&
      *cos(m(i)*zt(layer))*cos(m(j)*zt(layer)) - 2*m(j)*m(i)*1.0/(zb(&
      layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4&
      - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j&
      )**4)*at(layer)*cos(m(i)*zb(layer))*cos(m(j)*zb(layer)) + 2*m(j)*&
      m(i)*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(&
      layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2&
      - zt(layer)*m(j)**4)*at(layer)*cos(m(i)*zt(layer))*cos(m(j)*zt(&
      layer)) + m(j)*m(i)**2*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)&
      **2*m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)&
      *m(j)**2*m(i)**2 - zt(layer)*m(j)**4)*ab(layer)*zb(layer)*cos(m(j&
      )*zb(layer))*sin(m(i)*zb(layer)) - m(j)*m(i)**2*1.0/(zb(layer)*m(&
      i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(&
      layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)&
      *ab(layer)*zt(layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer)) - m(&
      j)*m(i)**2*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 +&
      zb(layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)&
      **2 - zt(layer)*m(j)**4)*at(layer)*zb(layer)*cos(m(j)*zb(layer))*&
      sin(m(i)*zb(layer)) + m(j)*m(i)**2*1.0/(zb(layer)*m(i)**4 - 2*zb(&
      layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 +&
      2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)*at(layer)*zt(&
      layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer)) + m(j)**2*1.0/(zb(&
      layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4&
      - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j&
      )**4)*ab(layer)*sin(m(i)*zb(layer))*sin(m(j)*zb(layer)) - m(j)**2&
      *1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(layer)&
      *m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(&
      layer)*m(j)**4)*ab(layer)*sin(m(i)*zt(layer))*sin(m(j)*zt(layer&
      )) - m(j)**2*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2&
      + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i&
      )**2 - zt(layer)*m(j)**4)*at(layer)*sin(m(i)*zb(layer))*sin(m(j)*&
      zb(layer)) + m(j)**2*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2&
      *m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(&
      j)**2*m(i)**2 - zt(layer)*m(j)**4)*at(layer)*sin(m(i)*zt(layer))*&
      sin(m(j)*zt(layer)) + m(j)**2*m(i)*1.0/(zb(layer)*m(i)**4 - 2*zb(&
      layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 +&
      2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)*ab(layer)*zb(&
      layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) - m(j)**2*m(i)*1.0&
      /(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j&
      )**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer&
      )*m(j)**4)*ab(layer)*zt(layer)*cos(m(i)*zt(layer))*sin(m(j)*zt(&
      layer)) - m(j)**2*m(i)*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)&
      **2*m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)&
      *m(j)**2*m(i)**2 - zt(layer)*m(j)**4)*at(layer)*zb(layer)*cos(m(i&
      )*zb(layer))*sin(m(j)*zb(layer)) + m(j)**2*m(i)*1.0/(zb(layer)*m(&
      i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(&
      layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)&
      *at(layer)*zt(layer)*cos(m(i)*zt(layer))*sin(m(j)*zt(layer)) - m(&
      j)**3*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(&
      layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2&
      - zt(layer)*m(j)**4)*ab(layer)*zb(layer)*cos(m(j)*zb(layer))*sin(&
      m(i)*zb(layer)) + m(j)**3*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(&
      j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(&
      layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)*ab(layer)*zt(layer)*&
      cos(m(j)*zt(layer))*sin(m(i)*zt(layer)) + m(j)**3*1.0/(zb(layer)*&
      m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(&
      layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)&
      *at(layer)*zb(layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) - m(&
      j)**3*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(&
      layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2&
      - zt(layer)*m(j)**4)*at(layer)*zt(layer)*cos(m(j)*zt(layer))*sin(&
      m(i)*zt(layer)) - zb(layer)*m(i)**3*1.0/(zb(layer)*m(i)**4 - 2*zb&
      (layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 +&
      2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)*at(layer)*cos(m(&
      i)*zb(layer))*sin(m(j)*zb(layer)) + zb(layer)*m(i)**3*1.0/(zb(&
      layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4&
      - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j&
      )**4)*at(layer)*cos(m(i)*zt(layer))*sin(m(j)*zt(layer)) + zb(&
      layer)*m(j)*m(i)**2*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*&
      m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j&
      )**2*m(i)**2 - zt(layer)*m(j)**4)*at(layer)*cos(m(j)*zb(layer))*&
      sin(m(i)*zb(layer)) - zb(layer)*m(j)*m(i)**2*1.0/(zb(layer)*m(i)&
      **4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(layer)&
      *m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)*at(&
      layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer)) + zb(layer)*m(j)**&
      2*m(i)*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(&
      layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2&
      - zt(layer)*m(j)**4)*at(layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(&
      layer)) - zb(layer)*m(j)**2*m(i)*1.0/(zb(layer)*m(i)**4 - 2*zb(&
      layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 +&
      2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)*at(layer)*cos(m(&
      i)*zt(layer))*sin(m(j)*zt(layer)) - zb(layer)*m(j)**3*1.0/(zb(&
      layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4&
      - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j&
      )**4)*at(layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) + zb(&
      layer)*m(j)**3*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)&
      **2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2&
      *m(i)**2 - zt(layer)*m(j)**4)*at(layer)*cos(m(j)*zt(layer))*sin(m&
      (i)*zt(layer)) + zt(layer)*m(i)**3*1.0/(zb(layer)*m(i)**4 - 2*zb(&
      layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 +&
      2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)*ab(layer)*cos(m(&
      i)*zb(layer))*sin(m(j)*zb(layer)) - zt(layer)*m(i)**3*1.0/(zb(&
      layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4&
      - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j&
      )**4)*ab(layer)*cos(m(i)*zt(layer))*sin(m(j)*zt(layer)) - zt(&
      layer)*m(j)*m(i)**2*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*&
      m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j&
      )**2*m(i)**2 - zt(layer)*m(j)**4)*ab(layer)*cos(m(j)*zb(layer))*&
      sin(m(i)*zb(layer)) + zt(layer)*m(j)*m(i)**2*1.0/(zb(layer)*m(i)&
      **4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(layer)&
      *m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)*ab(&
      layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer)) - zt(layer)*m(j)**&
      2*m(i)*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(&
      layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2&
      - zt(layer)*m(j)**4)*ab(layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(&
      layer)) + zt(layer)*m(j)**2*m(i)*1.0/(zb(layer)*m(i)**4 - 2*zb(&
      layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 +&
      2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)*ab(layer)*cos(m(&
      i)*zt(layer))*sin(m(j)*zt(layer)) + zt(layer)*m(j)**3*1.0/(zb(&
      layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4&
      - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j&
      )**4)*ab(layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) - zt(&
      layer)*m(j)**3*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)&
      **2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2&
      *m(i)**2 - zt(layer)*m(j)**4)*ab(layer)*cos(m(j)*zt(layer))*sin(m&
      (i)*zt(layer)))
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
        REAL(DP), intent(in), dimension(0:nlayers-1) :: at,ab,bt,bb, &
                                                        zt,zb
        REAL(DP), intent(out), dimension(0:neig-1, 0:neig-1) :: a
        INTEGER :: i , j, layer
        a=0.0D0
        DO layer = 0, nlayers-1
          DO j = 0, neig-1
              i=j
      a(i, i) = a(i, i) + (3*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)&
      *zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bb(layer)*ab(layer)&
      *cos(m(i)*zb(layer))*sin(m(i)*zb(layer)) - 3*1.0/(12*zb(layer)**2&
      *m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)&
      **3)*bb(layer)*ab(layer)*cos(m(i)*zt(layer))*sin(m(i)*zt(layer))&
      - 3*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3&
      + 12*zt(layer)**2*m(i)**3)*bb(layer)*at(layer)*cos(m(i)*zb(layer&
      ))*sin(m(i)*zb(layer)) + 3*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(&
      layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bb(layer)*at(&
      layer)*cos(m(i)*zt(layer))*sin(m(i)*zt(layer)) - 3*1.0/(12*zb(&
      layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)&
      **2*m(i)**3)*bt(layer)*ab(layer)*cos(m(i)*zb(layer))*sin(m(i)*zb(&
      layer)) + 3*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)&
      *m(i)**3 + 12*zt(layer)**2*m(i)**3)*bt(layer)*ab(layer)*cos(m(i)*&
      zt(layer))*sin(m(i)*zt(layer)) + 3*1.0/(12*zb(layer)**2*m(i)**3 -&
      24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bt(&
      layer)*at(layer)*cos(m(i)*zb(layer))*sin(m(i)*zb(layer)) - 3*1.0/&
      (12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt&
      (layer)**2*m(i)**3)*bt(layer)*at(layer)*cos(m(i)*zt(layer))*sin(m&
      (i)*zt(layer)) + 3*m(i)*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(&
      layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bb(layer)*ab(&
      layer)*zb(layer)*sin(m(i)*zb(layer))**2 - 3*m(i)*1.0/(12*zb(layer&
      )**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m&
      (i)**3)*bb(layer)*ab(layer)*zb(layer)*cos(m(i)*zb(layer))**2 - 3*&
      m(i)*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**&
      3 + 12*zt(layer)**2*m(i)**3)*bb(layer)*ab(layer)*zt(layer)*sin(m(&
      i)*zt(layer))**2 + 3*m(i)*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(&
      layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bb(layer)*ab(&
      layer)*zt(layer)*cos(m(i)*zt(layer))**2 - 3*m(i)*1.0/(12*zb(layer&
      )**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m&
      (i)**3)*bb(layer)*at(layer)*zb(layer)*sin(m(i)*zb(layer))**2 + 3*&
      m(i)*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**&
      3 + 12*zt(layer)**2*m(i)**3)*bb(layer)*at(layer)*zb(layer)*cos(m(&
      i)*zb(layer))**2 + 3*m(i)*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(&
      layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bb(layer)*at(&
      layer)*zt(layer)*sin(m(i)*zt(layer))**2 - 3*m(i)*1.0/(12*zb(layer&
      )**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m&
      (i)**3)*bb(layer)*at(layer)*zt(layer)*cos(m(i)*zt(layer))**2 - 3*&
      m(i)*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**&
      3 + 12*zt(layer)**2*m(i)**3)*bt(layer)*ab(layer)*zb(layer)*sin(m(&
      i)*zb(layer))**2 + 3*m(i)*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(&
      layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bt(layer)*ab(&
      layer)*zb(layer)*cos(m(i)*zb(layer))**2 + 3*m(i)*1.0/(12*zb(layer&
      )**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m&
      (i)**3)*bt(layer)*ab(layer)*zt(layer)*sin(m(i)*zt(layer))**2 - 3*&
      m(i)*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**&
      3 + 12*zt(layer)**2*m(i)**3)*bt(layer)*ab(layer)*zt(layer)*cos(m(&
      i)*zt(layer))**2 + 3*m(i)*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(&
      layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bt(layer)*at(&
      layer)*zb(layer)*sin(m(i)*zb(layer))**2 - 3*m(i)*1.0/(12*zb(layer&
      )**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m&
      (i)**3)*bt(layer)*at(layer)*zb(layer)*cos(m(i)*zb(layer))**2 - 3*&
      m(i)*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**&
      3 + 12*zt(layer)**2*m(i)**3)*bt(layer)*at(layer)*zt(layer)*sin(m(&
      i)*zt(layer))**2 + 3*m(i)*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(&
      layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bt(layer)*at(&
      layer)*zt(layer)*cos(m(i)*zt(layer))**2 - 6*m(i)**2*1.0/(12*zb(&
      layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)&
      **2*m(i)**3)*bb(layer)*ab(layer)*zb(layer)**2*cos(m(i)*zb(layer))&
      *sin(m(i)*zb(layer)) + 6*m(i)**2*1.0/(12*zb(layer)**2*m(i)**3 -&
      24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bb(&
      layer)*ab(layer)*zt(layer)**2*cos(m(i)*zt(layer))*sin(m(i)*zt(&
      layer)) + 6*m(i)**2*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*&
      zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bb(layer)*at(layer)*&
      zb(layer)**2*cos(m(i)*zb(layer))*sin(m(i)*zb(layer)) - 6*m(i)**2*&
      1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 +&
      12*zt(layer)**2*m(i)**3)*bb(layer)*at(layer)*zt(layer)**2*cos(m(i&
      )*zt(layer))*sin(m(i)*zt(layer)) + 6*m(i)**2*1.0/(12*zb(layer)**2&
      *m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)&
      **3)*bt(layer)*ab(layer)*zb(layer)**2*cos(m(i)*zb(layer))*sin(m(i&
      )*zb(layer)) - 6*m(i)**2*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(&
      layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bt(layer)*ab(&
      layer)*zt(layer)**2*cos(m(i)*zt(layer))*sin(m(i)*zt(layer)) - 6*m&
      (i)**2*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)&
      **3 + 12*zt(layer)**2*m(i)**3)*bt(layer)*at(layer)*zb(layer)**2*&
      cos(m(i)*zb(layer))*sin(m(i)*zb(layer)) + 6*m(i)**2*1.0/(12*zb(&
      layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)&
      **2*m(i)**3)*bt(layer)*at(layer)*zt(layer)**2*cos(m(i)*zt(layer))&
      *sin(m(i)*zt(layer)) + 2*m(i)**3*1.0/(12*zb(layer)**2*m(i)**3 -&
      24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bb(&
      layer)*ab(layer)*zb(layer)**3*sin(m(i)*zb(layer))**2 + 2*m(i)**3*&
      1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 +&
      12*zt(layer)**2*m(i)**3)*bb(layer)*ab(layer)*zb(layer)**3*cos(m(i&
      )*zb(layer))**2 - 2*m(i)**3*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(&
      layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bb(layer)*ab(&
      layer)*zt(layer)**3*sin(m(i)*zt(layer))**2 - 2*m(i)**3*1.0/(12*zb&
      (layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer&
      )**2*m(i)**3)*bb(layer)*ab(layer)*zt(layer)**3*cos(m(i)*zt(layer&
      ))**2 - 2*m(i)**3*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(&
      layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bb(layer)*at(layer)*zb(&
      layer)**3*sin(m(i)*zb(layer))**2 - 2*m(i)**3*1.0/(12*zb(layer)**2&
      *m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)&
      **3)*bb(layer)*at(layer)*zb(layer)**3*cos(m(i)*zb(layer))**2 + 2*&
      m(i)**3*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i&
      )**3 + 12*zt(layer)**2*m(i)**3)*bb(layer)*at(layer)*zt(layer)**3*&
      sin(m(i)*zt(layer))**2 + 2*m(i)**3*1.0/(12*zb(layer)**2*m(i)**3 -&
      24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bb(&
      layer)*at(layer)*zt(layer)**3*cos(m(i)*zt(layer))**2 - 2*m(i)**3*&
      1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 +&
      12*zt(layer)**2*m(i)**3)*bt(layer)*ab(layer)*zb(layer)**3*sin(m(i&
      )*zb(layer))**2 - 2*m(i)**3*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(&
      layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bt(layer)*ab(&
      layer)*zb(layer)**3*cos(m(i)*zb(layer))**2 + 2*m(i)**3*1.0/(12*zb&
      (layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer&
      )**2*m(i)**3)*bt(layer)*ab(layer)*zt(layer)**3*sin(m(i)*zt(layer&
      ))**2 + 2*m(i)**3*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(&
      layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bt(layer)*ab(layer)*zt(&
      layer)**3*cos(m(i)*zt(layer))**2 + 2*m(i)**3*1.0/(12*zb(layer)**2&
      *m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)&
      **3)*bt(layer)*at(layer)*zb(layer)**3*sin(m(i)*zb(layer))**2 + 2*&
      m(i)**3*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i&
      )**3 + 12*zt(layer)**2*m(i)**3)*bt(layer)*at(layer)*zb(layer)**3*&
      cos(m(i)*zb(layer))**2 - 2*m(i)**3*1.0/(12*zb(layer)**2*m(i)**3 -&
      24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bt(&
      layer)*at(layer)*zt(layer)**3*sin(m(i)*zt(layer))**2 - 2*m(i)**3*&
      1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 +&
      12*zt(layer)**2*m(i)**3)*bt(layer)*at(layer)*zt(layer)**3*cos(m(i&
      )*zt(layer))**2 - 3*zb(layer)*m(i)*1.0/(12*zb(layer)**2*m(i)**3 -&
      24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bb(&
      layer)*at(layer)*cos(m(i)*zb(layer))**2 + 3*zb(layer)*m(i)*1.0/(&
      12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(&
      layer)**2*m(i)**3)*bb(layer)*at(layer)*cos(m(i)*zt(layer))**2 - 3&
      *zb(layer)*m(i)*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(&
      layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bt(layer)*ab(layer)*cos&
      (m(i)*zb(layer))**2 + 3*zb(layer)*m(i)*1.0/(12*zb(layer)**2*m(i)&
      **3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*&
      bt(layer)*ab(layer)*cos(m(i)*zt(layer))**2 + 6*zb(layer)*m(i)*1.0&
      /(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*&
      zt(layer)**2*m(i)**3)*bt(layer)*at(layer)*cos(m(i)*zb(layer))**2&
      - 6*zb(layer)*m(i)*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb&
      (layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bt(layer)*at(layer)*&
      cos(m(i)*zt(layer))**2 - 6*zb(layer)*m(i)**2*1.0/(12*zb(layer)**2&
      *m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)&
      **3)*bb(layer)*at(layer)*zb(layer)*cos(m(i)*zb(layer))*sin(m(i)*&
      zb(layer)) + 6*zb(layer)*m(i)**2*1.0/(12*zb(layer)**2*m(i)**3 -&
      24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bb(&
      layer)*at(layer)*zt(layer)*cos(m(i)*zt(layer))*sin(m(i)*zt(layer&
      )) - 6*zb(layer)*m(i)**2*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(&
      layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bt(layer)*ab(&
      layer)*zb(layer)*cos(m(i)*zb(layer))*sin(m(i)*zb(layer)) + 6*zb(&
      layer)*m(i)**2*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(&
      layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bt(layer)*ab(layer)*zt(&
      layer)*cos(m(i)*zt(layer))*sin(m(i)*zt(layer)) + 12*zb(layer)*m(i&
      )**2*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**&
      3 + 12*zt(layer)**2*m(i)**3)*bt(layer)*at(layer)*zb(layer)*cos(m(&
      i)*zb(layer))*sin(m(i)*zb(layer)) - 12*zb(layer)*m(i)**2*1.0/(12*&
      zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(&
      layer)**2*m(i)**3)*bt(layer)*at(layer)*zt(layer)*cos(m(i)*zt(&
      layer))*sin(m(i)*zt(layer)) + 3*zb(layer)*m(i)**3*1.0/(12*zb(&
      layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)&
      **2*m(i)**3)*bb(layer)*at(layer)*zb(layer)**2*sin(m(i)*zb(layer))&
      **2 + 3*zb(layer)*m(i)**3*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(&
      layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bb(layer)*at(&
      layer)*zb(layer)**2*cos(m(i)*zb(layer))**2 - 3*zb(layer)*m(i)**3*&
      1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 +&
      12*zt(layer)**2*m(i)**3)*bb(layer)*at(layer)*zt(layer)**2*sin(m(i&
      )*zt(layer))**2 - 3*zb(layer)*m(i)**3*1.0/(12*zb(layer)**2*m(i)**&
      3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bb(&
      layer)*at(layer)*zt(layer)**2*cos(m(i)*zt(layer))**2 + 3*zb(layer&
      )*m(i)**3*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m&
      (i)**3 + 12*zt(layer)**2*m(i)**3)*bt(layer)*ab(layer)*zb(layer)**&
      2*sin(m(i)*zb(layer))**2 + 3*zb(layer)*m(i)**3*1.0/(12*zb(layer)&
      **2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(&
      i)**3)*bt(layer)*ab(layer)*zb(layer)**2*cos(m(i)*zb(layer))**2 -&
      3*zb(layer)*m(i)**3*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*&
      zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bt(layer)*ab(layer)*&
      zt(layer)**2*sin(m(i)*zt(layer))**2 - 3*zb(layer)*m(i)**3*1.0/(12&
      *zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(&
      layer)**2*m(i)**3)*bt(layer)*ab(layer)*zt(layer)**2*cos(m(i)*zt(&
      layer))**2 - 6*zb(layer)*m(i)**3*1.0/(12*zb(layer)**2*m(i)**3 -&
      24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bt(&
      layer)*at(layer)*zb(layer)**2*sin(m(i)*zb(layer))**2 - 6*zb(layer&
      )*m(i)**3*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m&
      (i)**3 + 12*zt(layer)**2*m(i)**3)*bt(layer)*at(layer)*zb(layer)**&
      2*cos(m(i)*zb(layer))**2 + 6*zb(layer)*m(i)**3*1.0/(12*zb(layer)&
      **2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(&
      i)**3)*bt(layer)*at(layer)*zt(layer)**2*sin(m(i)*zt(layer))**2 +&
      6*zb(layer)*m(i)**3*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*&
      zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bt(layer)*at(layer)*&
      zt(layer)**2*cos(m(i)*zt(layer))**2 - 6*zb(layer)**2*m(i)**2*1.0/&
      (12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt&
      (layer)**2*m(i)**3)*bt(layer)*at(layer)*cos(m(i)*zb(layer))*sin(m&
      (i)*zb(layer)) + 6*zb(layer)**2*m(i)**2*1.0/(12*zb(layer)**2*m(i)&
      **3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*&
      bt(layer)*at(layer)*cos(m(i)*zt(layer))*sin(m(i)*zt(layer)) + 6*&
      zb(layer)**2*m(i)**3*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*&
      zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bt(layer)*at(layer)*&
      zb(layer)*sin(m(i)*zb(layer))**2 + 6*zb(layer)**2*m(i)**3*1.0/(12&
      *zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(&
      layer)**2*m(i)**3)*bt(layer)*at(layer)*zb(layer)*cos(m(i)*zb(&
      layer))**2 - 6*zb(layer)**2*m(i)**3*1.0/(12*zb(layer)**2*m(i)**3&
      - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bt(&
      layer)*at(layer)*zt(layer)*sin(m(i)*zt(layer))**2 - 6*zb(layer)**&
      2*m(i)**3*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m&
      (i)**3 + 12*zt(layer)**2*m(i)**3)*bt(layer)*at(layer)*zt(layer)*&
      cos(m(i)*zt(layer))**2 + 6*zt(layer)*m(i)*1.0/(12*zb(layer)**2*m(&
      i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)&
      *bb(layer)*ab(layer)*cos(m(i)*zb(layer))**2 - 6*zt(layer)*m(i)*&
      1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 +&
      12*zt(layer)**2*m(i)**3)*bb(layer)*ab(layer)*cos(m(i)*zt(layer))&
      **2 - 3*zt(layer)*m(i)*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer&
      )*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bb(layer)*at(layer&
      )*cos(m(i)*zb(layer))**2 + 3*zt(layer)*m(i)*1.0/(12*zb(layer)**2*&
      m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**&
      3)*bb(layer)*at(layer)*cos(m(i)*zt(layer))**2 - 3*zt(layer)*m(i)*&
      1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 +&
      12*zt(layer)**2*m(i)**3)*bt(layer)*ab(layer)*cos(m(i)*zb(layer))&
      **2 + 3*zt(layer)*m(i)*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer&
      )*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bt(layer)*ab(layer&
      )*cos(m(i)*zt(layer))**2 + 12*zt(layer)*m(i)**2*1.0/(12*zb(layer)&
      **2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(&
      i)**3)*bb(layer)*ab(layer)*zb(layer)*cos(m(i)*zb(layer))*sin(m(i)&
      *zb(layer)) - 12*zt(layer)*m(i)**2*1.0/(12*zb(layer)**2*m(i)**3 -&
      24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bb(&
      layer)*ab(layer)*zt(layer)*cos(m(i)*zt(layer))*sin(m(i)*zt(layer&
      )) - 6*zt(layer)*m(i)**2*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(&
      layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bb(layer)*at(&
      layer)*zb(layer)*cos(m(i)*zb(layer))*sin(m(i)*zb(layer)) + 6*zt(&
      layer)*m(i)**2*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(&
      layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bb(layer)*at(layer)*zt(&
      layer)*cos(m(i)*zt(layer))*sin(m(i)*zt(layer)) - 6*zt(layer)*m(i)&
      **2*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3&
      + 12*zt(layer)**2*m(i)**3)*bt(layer)*ab(layer)*zb(layer)*cos(m(i)&
      *zb(layer))*sin(m(i)*zb(layer)) + 6*zt(layer)*m(i)**2*1.0/(12*zb(&
      layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)&
      **2*m(i)**3)*bt(layer)*ab(layer)*zt(layer)*cos(m(i)*zt(layer))*&
      sin(m(i)*zt(layer)) - 6*zt(layer)*m(i)**3*1.0/(12*zb(layer)**2*m(&
      i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)&
      *bb(layer)*ab(layer)*zb(layer)**2*sin(m(i)*zb(layer))**2 - 6*zt(&
      layer)*m(i)**3*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(&
      layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bb(layer)*ab(layer)*zb(&
      layer)**2*cos(m(i)*zb(layer))**2 + 6*zt(layer)*m(i)**3*1.0/(12*zb&
      (layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer&
      )**2*m(i)**3)*bb(layer)*ab(layer)*zt(layer)**2*sin(m(i)*zt(layer&
      ))**2 + 6*zt(layer)*m(i)**3*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(&
      layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bb(layer)*ab(&
      layer)*zt(layer)**2*cos(m(i)*zt(layer))**2 + 3*zt(layer)*m(i)**3*&
      1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 +&
      12*zt(layer)**2*m(i)**3)*bb(layer)*at(layer)*zb(layer)**2*sin(m(i&
      )*zb(layer))**2 + 3*zt(layer)*m(i)**3*1.0/(12*zb(layer)**2*m(i)**&
      3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bb(&
      layer)*at(layer)*zb(layer)**2*cos(m(i)*zb(layer))**2 - 3*zt(layer&
      )*m(i)**3*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m&
      (i)**3 + 12*zt(layer)**2*m(i)**3)*bb(layer)*at(layer)*zt(layer)**&
      2*sin(m(i)*zt(layer))**2 - 3*zt(layer)*m(i)**3*1.0/(12*zb(layer)&
      **2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(&
      i)**3)*bb(layer)*at(layer)*zt(layer)**2*cos(m(i)*zt(layer))**2 +&
      3*zt(layer)*m(i)**3*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*&
      zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bt(layer)*ab(layer)*&
      zb(layer)**2*sin(m(i)*zb(layer))**2 + 3*zt(layer)*m(i)**3*1.0/(12&
      *zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(&
      layer)**2*m(i)**3)*bt(layer)*ab(layer)*zb(layer)**2*cos(m(i)*zb(&
      layer))**2 - 3*zt(layer)*m(i)**3*1.0/(12*zb(layer)**2*m(i)**3 -&
      24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bt(&
      layer)*ab(layer)*zt(layer)**2*sin(m(i)*zt(layer))**2 - 3*zt(layer&
      )*m(i)**3*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m&
      (i)**3 + 12*zt(layer)**2*m(i)**3)*bt(layer)*ab(layer)*zt(layer)**&
      2*cos(m(i)*zt(layer))**2 + 6*zt(layer)*zb(layer)*m(i)**2*1.0/(12*&
      zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(&
      layer)**2*m(i)**3)*bb(layer)*at(layer)*cos(m(i)*zb(layer))*sin(m(&
      i)*zb(layer)) - 6*zt(layer)*zb(layer)*m(i)**2*1.0/(12*zb(layer)**&
      2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)&
      **3)*bb(layer)*at(layer)*cos(m(i)*zt(layer))*sin(m(i)*zt(layer))&
      + 6*zt(layer)*zb(layer)*m(i)**2*1.0/(12*zb(layer)**2*m(i)**3 - 24&
      *zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bt(layer)&
      *ab(layer)*cos(m(i)*zb(layer))*sin(m(i)*zb(layer)) - 6*zt(layer)*&
      zb(layer)*m(i)**2*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(&
      layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bt(layer)*ab(layer)*cos&
      (m(i)*zt(layer))*sin(m(i)*zt(layer)) - 6*zt(layer)*zb(layer)*m(i)&
      **3*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3&
      + 12*zt(layer)**2*m(i)**3)*bb(layer)*at(layer)*zb(layer)*sin(m(i)&
      *zb(layer))**2 - 6*zt(layer)*zb(layer)*m(i)**3*1.0/(12*zb(layer)&
      **2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(&
      i)**3)*bb(layer)*at(layer)*zb(layer)*cos(m(i)*zb(layer))**2 + 6*&
      zt(layer)*zb(layer)*m(i)**3*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(&
      layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bb(layer)*at(&
      layer)*zt(layer)*sin(m(i)*zt(layer))**2 + 6*zt(layer)*zb(layer)*m&
      (i)**3*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)&
      **3 + 12*zt(layer)**2*m(i)**3)*bb(layer)*at(layer)*zt(layer)*cos(&
      m(i)*zt(layer))**2 - 6*zt(layer)*zb(layer)*m(i)**3*1.0/(12*zb(&
      layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)&
      **2*m(i)**3)*bt(layer)*ab(layer)*zb(layer)*sin(m(i)*zb(layer))**2&
      - 6*zt(layer)*zb(layer)*m(i)**3*1.0/(12*zb(layer)**2*m(i)**3 - 24&
      *zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bt(layer)&
      *ab(layer)*zb(layer)*cos(m(i)*zb(layer))**2 + 6*zt(layer)*zb(&
      layer)*m(i)**3*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(&
      layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bt(layer)*ab(layer)*zt(&
      layer)*sin(m(i)*zt(layer))**2 + 6*zt(layer)*zb(layer)*m(i)**3*1.0&
      /(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*&
      zt(layer)**2*m(i)**3)*bt(layer)*ab(layer)*zt(layer)*cos(m(i)*zt(&
      layer))**2 - 6*zt(layer)**2*m(i)**2*1.0/(12*zb(layer)**2*m(i)**3&
      - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bb(&
      layer)*ab(layer)*cos(m(i)*zb(layer))*sin(m(i)*zb(layer)) + 6*zt(&
      layer)**2*m(i)**2*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(&
      layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bb(layer)*ab(layer)*cos&
      (m(i)*zt(layer))*sin(m(i)*zt(layer)) + 6*zt(layer)**2*m(i)**3*1.0&
      /(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*&
      zt(layer)**2*m(i)**3)*bb(layer)*ab(layer)*zb(layer)*sin(m(i)*zb(&
      layer))**2 + 6*zt(layer)**2*m(i)**3*1.0/(12*zb(layer)**2*m(i)**3&
      - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(i)**3)*bb(&
      layer)*ab(layer)*zb(layer)*cos(m(i)*zb(layer))**2 - 6*zt(layer)**&
      2*m(i)**3*1.0/(12*zb(layer)**2*m(i)**3 - 24*zt(layer)*zb(layer)*m&
      (i)**3 + 12*zt(layer)**2*m(i)**3)*bb(layer)*ab(layer)*zt(layer)*&
      sin(m(i)*zt(layer))**2 - 6*zt(layer)**2*m(i)**3*1.0/(12*zb(layer)&
      **2*m(i)**3 - 24*zt(layer)*zb(layer)*m(i)**3 + 12*zt(layer)**2*m(&
      i)**3)*bb(layer)*ab(layer)*zt(layer)*cos(m(i)*zt(layer))**2)
            DO i = j+1, neig-1
      a(i, j) = a(i, j) + (2*m(i)**3*1.0/(zb(layer)**2*m(i)**6 - 3*zb(&
      layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bb(layer)*ab(layer)*cos(m(i)*zb(layer))*sin(m(&
      j)*zb(layer)) - 2*m(i)**3*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)&
      **2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)&
      **2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(&
      layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 +&
      2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer&
      )**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)&
      **2*m(j)**6)*bb(layer)*ab(layer)*cos(m(i)*zt(layer))*sin(m(j)*zt(&
      layer)) - 2*m(i)**3*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(&
      j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j&
      )**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)&
      **2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)&
      *zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**&
      2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6&
      )*bb(layer)*at(layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) + 2&
      *m(i)**3*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)&
      **4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*&
      zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)&
      **4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb&
      (layer)*at(layer)*cos(m(i)*zt(layer))*sin(m(j)*zt(layer)) - 2*m(i&
      )**3*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 +&
      3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*&
      ab(layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) + 2*m(i)**3*1.0&
      /(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(&
      layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb&
      (layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(&
      layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(&
      layer)*cos(m(i)*zt(layer))*sin(m(j)*zt(layer)) + 2*m(i)**3*1.0/(&
      zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(&
      layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb&
      (layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(&
      layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*at(&
      layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) - 2*m(i)**3*1.0/(&
      zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(&
      layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb&
      (layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(&
      layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*at(&
      layer)*cos(m(i)*zt(layer))*sin(m(j)*zt(layer)) + 2*m(i)**4*1.0/(&
      zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(&
      layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb&
      (layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(&
      layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*ab(&
      layer)*zb(layer)*sin(m(i)*zb(layer))*sin(m(j)*zb(layer)) - 2*m(i)&
      **4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 +&
      3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*&
      ab(layer)*zt(layer)*sin(m(i)*zt(layer))*sin(m(j)*zt(layer)) - 2*m&
      (i)**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4&
      + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*&
      at(layer)*zb(layer)*sin(m(i)*zb(layer))*sin(m(j)*zb(layer)) + 2*m&
      (i)**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4&
      + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*&
      at(layer)*zt(layer)*sin(m(i)*zt(layer))*sin(m(j)*zt(layer)) - 2*m&
      (i)**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4&
      + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*&
      ab(layer)*zb(layer)*sin(m(i)*zb(layer))*sin(m(j)*zb(layer)) + 2*m&
      (i)**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4&
      + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*&
      ab(layer)*zt(layer)*sin(m(i)*zt(layer))*sin(m(j)*zt(layer)) + 2*m&
      (i)**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4&
      + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*&
      at(layer)*zb(layer)*sin(m(i)*zb(layer))*sin(m(j)*zb(layer)) - 2*m&
      (i)**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4&
      + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*&
      at(layer)*zt(layer)*sin(m(i)*zt(layer))*sin(m(j)*zt(layer)) - m(i&
      )**5*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 +&
      3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*&
      ab(layer)*zb(layer)**2*cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) +&
      m(i)**5*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**&
      4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*&
      ab(layer)*zt(layer)**2*cos(m(i)*zt(layer))*sin(m(j)*zt(layer)) +&
      m(i)**5*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**&
      4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*&
      at(layer)*zb(layer)**2*cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) -&
      m(i)**5*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**&
      4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*&
      at(layer)*zt(layer)**2*cos(m(i)*zt(layer))*sin(m(j)*zt(layer)) +&
      m(i)**5*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**&
      4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*&
      ab(layer)*zb(layer)**2*cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) -&
      m(i)**5*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**&
      4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*&
      ab(layer)*zt(layer)**2*cos(m(i)*zt(layer))*sin(m(j)*zt(layer)) -&
      m(i)**5*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**&
      4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*&
      at(layer)*zb(layer)**2*cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) +&
      m(i)**5*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**&
      4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*&
      at(layer)*zt(layer)**2*cos(m(i)*zt(layer))*sin(m(j)*zt(layer)) -&
      6*m(j)*m(i)**2*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2&
      *m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6&
      - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m&
      (i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb&
      (layer)*ab(layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) + 6*m(j&
      )*m(i)**2*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)&
      **4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*&
      zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)&
      **4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb&
      (layer)*ab(layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer)) + 6*m(j&
      )*m(i)**2*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)&
      **4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*&
      zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)&
      **4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb&
      (layer)*at(layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) - 6*m(j&
      )*m(i)**2*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)&
      **4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*&
      zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)&
      **4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb&
      (layer)*at(layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer)) + 6*m(j&
      )*m(i)**2*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)&
      **4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*&
      zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)&
      **4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt&
      (layer)*ab(layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) - 6*m(j&
      )*m(i)**2*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)&
      **4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*&
      zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)&
      **4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt&
      (layer)*ab(layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer)) - 6*m(j&
      )*m(i)**2*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)&
      **4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*&
      zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)&
      **4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt&
      (layer)*at(layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) + 6*m(j&
      )*m(i)**2*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)&
      **4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*&
      zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)&
      **4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt&
      (layer)*at(layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer)) + 4*m(j&
      )*m(i)**3*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)&
      **4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*&
      zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)&
      **4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb&
      (layer)*ab(layer)*zb(layer)*cos(m(i)*zb(layer))*cos(m(j)*zb(layer&
      )) - 4*m(j)*m(i)**3*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(&
      j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j&
      )**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)&
      **2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)&
      *zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**&
      2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6&
      )*bb(layer)*ab(layer)*zt(layer)*cos(m(i)*zt(layer))*cos(m(j)*zt(&
      layer)) - 4*m(j)*m(i)**3*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)&
      **2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)&
      **2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(&
      layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 +&
      2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer&
      )**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)&
      **2*m(j)**6)*bb(layer)*at(layer)*zb(layer)*cos(m(i)*zb(layer))*&
      cos(m(j)*zb(layer)) + 4*m(j)*m(i)**3*1.0/(zb(layer)**2*m(i)**6 -&
      3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 -&
      zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer&
      )*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)&
      **2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt&
      (layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bb(layer)*at(layer)*zt(layer)*cos(m(i)*zt(&
      layer))*cos(m(j)*zt(layer)) - 4*m(j)*m(i)**3*1.0/(zb(layer)**2*m(&
      i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m&
      (i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6&
      *zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)&
      **4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**&
      6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)&
      **2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(layer)*zb(layer)*cos(m(i&
      )*zb(layer))*cos(m(j)*zb(layer)) + 4*m(j)*m(i)**3*1.0/(zb(layer)&
      **2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j&
      )**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)&
      **6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(&
      layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer&
      )**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(&
      j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(layer)*zt(&
      layer)*cos(m(i)*zt(layer))*cos(m(j)*zt(layer)) + 4*m(j)*m(i)**3*&
      1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb&
      (layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*&
      zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(&
      layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*at(&
      layer)*zb(layer)*cos(m(i)*zb(layer))*cos(m(j)*zb(layer)) - 4*m(j)&
      *m(i)**3*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)&
      **4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*&
      zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)&
      **4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt&
      (layer)*at(layer)*zt(layer)*cos(m(i)*zt(layer))*cos(m(j)*zt(layer&
      )) + m(j)*m(i)**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)&
      **2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)&
      **6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)&
      **2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)&
      *zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**&
      2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6&
      )*bb(layer)*ab(layer)*zb(layer)**2*cos(m(j)*zb(layer))*sin(m(i)*&
      zb(layer)) - m(j)*m(i)**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)&
      **2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)&
      **2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(&
      layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 +&
      2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer&
      )**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)&
      **2*m(j)**6)*bb(layer)*ab(layer)*zt(layer)**2*cos(m(j)*zt(layer))&
      *sin(m(i)*zt(layer)) - m(j)*m(i)**4*1.0/(zb(layer)**2*m(i)**6 - 3&
      *zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 -&
      zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer&
      )*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)&
      **2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt&
      (layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bb(layer)*at(layer)*zb(layer)**2*cos(m(j)*zb(&
      layer))*sin(m(i)*zb(layer)) + m(j)*m(i)**4*1.0/(zb(layer)**2*m(i)&
      **6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i&
      )**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*&
      zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)&
      **4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**&
      6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)&
      **2 - zt(layer)**2*m(j)**6)*bb(layer)*at(layer)*zt(layer)**2*cos(&
      m(j)*zt(layer))*sin(m(i)*zt(layer)) - m(j)*m(i)**4*1.0/(zb(layer)&
      **2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j&
      )**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)&
      **6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(&
      layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer&
      )**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(&
      j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(layer)*zb(&
      layer)**2*cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) + m(j)*m(i)**4*&
      1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb&
      (layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*&
      zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(&
      layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(&
      layer)*zt(layer)**2*cos(m(j)*zt(layer))*sin(m(i)*zt(layer)) + m(j&
      )*m(i)**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)&
      **4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*&
      zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)&
      **4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt&
      (layer)*at(layer)*zb(layer)**2*cos(m(j)*zb(layer))*sin(m(i)*zb(&
      layer)) - m(j)*m(i)**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2&
      *m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*&
      m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m&
      (j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(&
      layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*&
      m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m&
      (j)**6)*bt(layer)*at(layer)*zt(layer)**2*cos(m(j)*zt(layer))*sin(&
      m(i)*zt(layer)) + 6*m(j)**2*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb&
      (layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bb(layer)*ab(layer)*cos(m(i)*zb(layer))*sin(m(&
      j)*zb(layer)) - 6*m(j)**2*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(&
      layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bb(layer)*ab(layer)*cos(m(i)*zt(layer))*sin(m(&
      j)*zt(layer)) - 6*m(j)**2*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(&
      layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bb(layer)*at(layer)*cos(m(i)*zb(layer))*sin(m(&
      j)*zb(layer)) + 6*m(j)**2*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(&
      layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bb(layer)*at(layer)*cos(m(i)*zt(layer))*sin(m(&
      j)*zt(layer)) - 6*m(j)**2*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(&
      layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bt(layer)*ab(layer)*cos(m(i)*zb(layer))*sin(m(&
      j)*zb(layer)) + 6*m(j)**2*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(&
      layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bt(layer)*ab(layer)*cos(m(i)*zt(layer))*sin(m(&
      j)*zt(layer)) + 6*m(j)**2*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(&
      layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bt(layer)*at(layer)*cos(m(i)*zb(layer))*sin(m(&
      j)*zb(layer)) - 6*m(j)**2*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(&
      layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bt(layer)*at(layer)*cos(m(i)*zt(layer))*sin(m(&
      j)*zt(layer)) + 2*m(j)**2*m(i)**3*1.0/(zb(layer)**2*m(i)**6 - 3*&
      zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 -&
      zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer&
      )*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)&
      **2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt&
      (layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bb(layer)*ab(layer)*zb(layer)**2*cos(m(i)*zb(&
      layer))*sin(m(j)*zb(layer)) - 2*m(j)**2*m(i)**3*1.0/(zb(layer)**2&
      *m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**&
      4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6&
      + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m&
      (j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i&
      )**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(&
      i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*ab(layer)*zt(layer)**2*&
      cos(m(i)*zt(layer))*sin(m(j)*zt(layer)) - 2*m(j)**2*m(i)**3*1.0/(&
      zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(&
      layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb&
      (layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(&
      layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*at(&
      layer)*zb(layer)**2*cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) + 2*m&
      (j)**2*m(i)**3*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2&
      *m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6&
      - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m&
      (i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb&
      (layer)*at(layer)*zt(layer)**2*cos(m(i)*zt(layer))*sin(m(j)*zt(&
      layer)) - 2*m(j)**2*m(i)**3*1.0/(zb(layer)**2*m(i)**6 - 3*zb(&
      layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bt(layer)*ab(layer)*zb(layer)**2*cos(m(i)*zb(&
      layer))*sin(m(j)*zb(layer)) + 2*m(j)**2*m(i)**3*1.0/(zb(layer)**2&
      *m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**&
      4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6&
      + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m&
      (j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i&
      )**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(&
      i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(layer)*zt(layer)**2*&
      cos(m(i)*zt(layer))*sin(m(j)*zt(layer)) + 2*m(j)**2*m(i)**3*1.0/(&
      zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(&
      layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb&
      (layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(&
      layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*at(&
      layer)*zb(layer)**2*cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) - 2*m&
      (j)**2*m(i)**3*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2&
      *m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6&
      - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m&
      (i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt&
      (layer)*at(layer)*zt(layer)**2*cos(m(i)*zt(layer))*sin(m(j)*zt(&
      layer)) - 2*m(j)**3*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(&
      j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j&
      )**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)&
      **2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)&
      *zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**&
      2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6&
      )*bb(layer)*ab(layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) + 2&
      *m(j)**3*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)&
      **4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*&
      zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)&
      **4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb&
      (layer)*ab(layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer)) + 2*m(j&
      )**3*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 +&
      3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*&
      at(layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) - 2*m(j)**3*1.0&
      /(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(&
      layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb&
      (layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(&
      layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*at(&
      layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer)) + 2*m(j)**3*1.0/(&
      zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(&
      layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb&
      (layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(&
      layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(&
      layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) - 2*m(j)**3*1.0/(&
      zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(&
      layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb&
      (layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(&
      layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(&
      layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer)) - 2*m(j)**3*1.0/(&
      zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(&
      layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb&
      (layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(&
      layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*at(&
      layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) + 2*m(j)**3*1.0/(&
      zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(&
      layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb&
      (layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(&
      layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*at(&
      layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer)) - 4*m(j)**3*m(i)*&
      1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb&
      (layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*&
      zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(&
      layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*ab(&
      layer)*zb(layer)*cos(m(i)*zb(layer))*cos(m(j)*zb(layer)) + 4*m(j)&
      **3*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)&
      **4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*&
      zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)&
      **4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb&
      (layer)*ab(layer)*zt(layer)*cos(m(i)*zt(layer))*cos(m(j)*zt(layer&
      )) + 4*m(j)**3*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(&
      j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j&
      )**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)&
      **2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)&
      *zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**&
      2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6&
      )*bb(layer)*at(layer)*zb(layer)*cos(m(i)*zb(layer))*cos(m(j)*zb(&
      layer)) - 4*m(j)**3*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)&
      **2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)&
      **2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(&
      layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 +&
      2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer&
      )**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)&
      **2*m(j)**6)*bb(layer)*at(layer)*zt(layer)*cos(m(i)*zt(layer))*&
      cos(m(j)*zt(layer)) + 4*m(j)**3*m(i)*1.0/(zb(layer)**2*m(i)**6 -&
      3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 -&
      zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer&
      )*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)&
      **2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt&
      (layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bt(layer)*ab(layer)*zb(layer)*cos(m(i)*zb(&
      layer))*cos(m(j)*zb(layer)) - 4*m(j)**3*m(i)*1.0/(zb(layer)**2*m(&
      i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m&
      (i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6&
      *zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)&
      **4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**&
      6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)&
      **2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(layer)*zt(layer)*cos(m(i&
      )*zt(layer))*cos(m(j)*zt(layer)) - 4*m(j)**3*m(i)*1.0/(zb(layer)&
      **2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j&
      )**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)&
      **6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(&
      layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer&
      )**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(&
      j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*at(layer)*zb(&
      layer)*cos(m(i)*zb(layer))*cos(m(j)*zb(layer)) + 4*m(j)**3*m(i)*&
      1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb&
      (layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*&
      zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(&
      layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*at(&
      layer)*zt(layer)*cos(m(i)*zt(layer))*cos(m(j)*zt(layer)) - 2*m(j)&
      **3*m(i)**2*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(&
      i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2&
      *zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)&
      **4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb&
      (layer)*ab(layer)*zb(layer)**2*cos(m(j)*zb(layer))*sin(m(i)*zb(&
      layer)) + 2*m(j)**3*m(i)**2*1.0/(zb(layer)**2*m(i)**6 - 3*zb(&
      layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bb(layer)*ab(layer)*zt(layer)**2*cos(m(j)*zt(&
      layer))*sin(m(i)*zt(layer)) + 2*m(j)**3*m(i)**2*1.0/(zb(layer)**2&
      *m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**&
      4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6&
      + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m&
      (j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i&
      )**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(&
      i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*at(layer)*zb(layer)**2*&
      cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) - 2*m(j)**3*m(i)**2*1.0/(&
      zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(&
      layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb&
      (layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(&
      layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*at(&
      layer)*zt(layer)**2*cos(m(j)*zt(layer))*sin(m(i)*zt(layer)) + 2*m&
      (j)**3*m(i)**2*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2&
      *m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6&
      - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m&
      (i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt&
      (layer)*ab(layer)*zb(layer)**2*cos(m(j)*zb(layer))*sin(m(i)*zb(&
      layer)) - 2*m(j)**3*m(i)**2*1.0/(zb(layer)**2*m(i)**6 - 3*zb(&
      layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bt(layer)*ab(layer)*zt(layer)**2*cos(m(j)*zt(&
      layer))*sin(m(i)*zt(layer)) - 2*m(j)**3*m(i)**2*1.0/(zb(layer)**2&
      *m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**&
      4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6&
      + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m&
      (j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i&
      )**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(&
      i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*at(layer)*zb(layer)**2*&
      cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) + 2*m(j)**3*m(i)**2*1.0/(&
      zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(&
      layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb&
      (layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(&
      layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*at(&
      layer)*zt(layer)**2*cos(m(j)*zt(layer))*sin(m(i)*zt(layer)) - 2*m&
      (j)**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4&
      + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*&
      ab(layer)*zb(layer)*sin(m(i)*zb(layer))*sin(m(j)*zb(layer)) + 2*m&
      (j)**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4&
      + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*&
      ab(layer)*zt(layer)*sin(m(i)*zt(layer))*sin(m(j)*zt(layer)) + 2*m&
      (j)**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4&
      + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*&
      at(layer)*zb(layer)*sin(m(i)*zb(layer))*sin(m(j)*zb(layer)) - 2*m&
      (j)**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4&
      + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*&
      at(layer)*zt(layer)*sin(m(i)*zt(layer))*sin(m(j)*zt(layer)) + 2*m&
      (j)**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4&
      + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*&
      ab(layer)*zb(layer)*sin(m(i)*zb(layer))*sin(m(j)*zb(layer)) - 2*m&
      (j)**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4&
      + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*&
      ab(layer)*zt(layer)*sin(m(i)*zt(layer))*sin(m(j)*zt(layer)) - 2*m&
      (j)**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4&
      + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*&
      at(layer)*zb(layer)*sin(m(i)*zb(layer))*sin(m(j)*zb(layer)) + 2*m&
      (j)**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4&
      + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*&
      at(layer)*zt(layer)*sin(m(i)*zt(layer))*sin(m(j)*zt(layer)) - m(j&
      )**4*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)&
      **4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*&
      zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)&
      **4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb&
      (layer)*ab(layer)*zb(layer)**2*cos(m(i)*zb(layer))*sin(m(j)*zb(&
      layer)) + m(j)**4*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2&
      *m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*&
      m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m&
      (j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(&
      layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*&
      m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m&
      (j)**6)*bb(layer)*ab(layer)*zt(layer)**2*cos(m(i)*zt(layer))*sin(&
      m(j)*zt(layer)) + m(j)**4*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(&
      layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bb(layer)*at(layer)*zb(layer)**2*cos(m(i)*zb(&
      layer))*sin(m(j)*zb(layer)) - m(j)**4*m(i)*1.0/(zb(layer)**2*m(i)&
      **6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i&
      )**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*&
      zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)&
      **4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**&
      6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)&
      **2 - zt(layer)**2*m(j)**6)*bb(layer)*at(layer)*zt(layer)**2*cos(&
      m(i)*zt(layer))*sin(m(j)*zt(layer)) + m(j)**4*m(i)*1.0/(zb(layer)&
      **2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j&
      )**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)&
      **6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(&
      layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer&
      )**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(&
      j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(layer)*zb(&
      layer)**2*cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) - m(j)**4*m(i)*&
      1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb&
      (layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*&
      zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(&
      layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(&
      layer)*zt(layer)**2*cos(m(i)*zt(layer))*sin(m(j)*zt(layer)) - m(j&
      )**4*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)&
      **4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*&
      zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)&
      **4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt&
      (layer)*at(layer)*zb(layer)**2*cos(m(i)*zb(layer))*sin(m(j)*zb(&
      layer)) + m(j)**4*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2&
      *m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*&
      m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m&
      (j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(&
      layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*&
      m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m&
      (j)**6)*bt(layer)*at(layer)*zt(layer)**2*cos(m(i)*zt(layer))*sin(&
      m(j)*zt(layer)) + m(j)**5*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)&
      **2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)&
      **2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(&
      layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 +&
      2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer&
      )**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)&
      **2*m(j)**6)*bb(layer)*ab(layer)*zb(layer)**2*cos(m(j)*zb(layer))&
      *sin(m(i)*zb(layer)) - m(j)**5*1.0/(zb(layer)**2*m(i)**6 - 3*zb(&
      layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bb(layer)*ab(layer)*zt(layer)**2*cos(m(j)*zt(&
      layer))*sin(m(i)*zt(layer)) - m(j)**5*1.0/(zb(layer)**2*m(i)**6 -&
      3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 -&
      zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer&
      )*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)&
      **2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt&
      (layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bb(layer)*at(layer)*zb(layer)**2*cos(m(j)*zb(&
      layer))*sin(m(i)*zb(layer)) + m(j)**5*1.0/(zb(layer)**2*m(i)**6 -&
      3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 -&
      zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer&
      )*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)&
      **2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt&
      (layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bb(layer)*at(layer)*zt(layer)**2*cos(m(j)*zt(&
      layer))*sin(m(i)*zt(layer)) - m(j)**5*1.0/(zb(layer)**2*m(i)**6 -&
      3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 -&
      zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer&
      )*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)&
      **2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt&
      (layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bt(layer)*ab(layer)*zb(layer)**2*cos(m(j)*zb(&
      layer))*sin(m(i)*zb(layer)) + m(j)**5*1.0/(zb(layer)**2*m(i)**6 -&
      3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 -&
      zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer&
      )*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)&
      **2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt&
      (layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bt(layer)*ab(layer)*zt(layer)**2*cos(m(j)*zt(&
      layer))*sin(m(i)*zt(layer)) + m(j)**5*1.0/(zb(layer)**2*m(i)**6 -&
      3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 -&
      zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer&
      )*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)&
      **2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt&
      (layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bt(layer)*at(layer)*zb(layer)**2*cos(m(j)*zb(&
      layer))*sin(m(i)*zb(layer)) - m(j)**5*1.0/(zb(layer)**2*m(i)**6 -&
      3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 -&
      zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer&
      )*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)&
      **2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt&
      (layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bt(layer)*at(layer)*zt(layer)**2*cos(m(j)*zt(&
      layer))*sin(m(i)*zt(layer)) + zb(layer)*m(i)**4*1.0/(zb(layer)**2&
      *m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**&
      4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6&
      + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m&
      (j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i&
      )**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(&
      i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*at(layer)*sin(m(i)*zb(&
      layer))*sin(m(j)*zb(layer)) - zb(layer)*m(i)**4*1.0/(zb(layer)**2&
      *m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**&
      4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6&
      + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m&
      (j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i&
      )**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(&
      i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*at(layer)*sin(m(i)*zt(&
      layer))*sin(m(j)*zt(layer)) + zb(layer)*m(i)**4*1.0/(zb(layer)**2&
      *m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**&
      4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6&
      + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m&
      (j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i&
      )**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(&
      i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(layer)*sin(m(i)*zb(&
      layer))*sin(m(j)*zb(layer)) - zb(layer)*m(i)**4*1.0/(zb(layer)**2&
      *m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**&
      4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6&
      + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m&
      (j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i&
      )**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(&
      i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(layer)*sin(m(i)*zt(&
      layer))*sin(m(j)*zt(layer)) - 2*zb(layer)*m(i)**4*1.0/(zb(layer)&
      **2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j&
      )**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)&
      **6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(&
      layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer&
      )**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(&
      j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*at(layer)*sin(m(i&
      )*zb(layer))*sin(m(j)*zb(layer)) + 2*zb(layer)*m(i)**4*1.0/(zb(&
      layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)&
      **2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer&
      )*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*&
      zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(&
      layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)&
      **2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*at(layer)*&
      sin(m(i)*zt(layer))*sin(m(j)*zt(layer)) - zb(layer)*m(i)**5*1.0/(&
      zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(&
      layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb&
      (layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(&
      layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*at(&
      layer)*zb(layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) + zb(&
      layer)*m(i)**5*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2&
      *m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6&
      - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m&
      (i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb&
      (layer)*at(layer)*zt(layer)*cos(m(i)*zt(layer))*sin(m(j)*zt(layer&
      )) - zb(layer)*m(i)**5*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2&
      *m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*&
      m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m&
      (j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(&
      layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*&
      m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m&
      (j)**6)*bt(layer)*ab(layer)*zb(layer)*cos(m(i)*zb(layer))*sin(m(j&
      )*zb(layer)) + zb(layer)*m(i)**5*1.0/(zb(layer)**2*m(i)**6 - 3*zb&
      (layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bt(layer)*ab(layer)*zt(layer)*cos(m(i)*zt(&
      layer))*sin(m(j)*zt(layer)) + 2*zb(layer)*m(i)**5*1.0/(zb(layer)&
      **2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j&
      )**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)&
      **6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(&
      layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer&
      )**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(&
      j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*at(layer)*zb(&
      layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) - 2*zb(layer)*m(i)&
      **5*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 +&
      3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*&
      at(layer)*zt(layer)*cos(m(i)*zt(layer))*sin(m(j)*zt(layer)) + 2*&
      zb(layer)*m(j)*m(i)**3*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2&
      *m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*&
      m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m&
      (j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(&
      layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*&
      m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m&
      (j)**6)*bb(layer)*at(layer)*cos(m(i)*zb(layer))*cos(m(j)*zb(layer&
      )) - 2*zb(layer)*m(j)*m(i)**3*1.0/(zb(layer)**2*m(i)**6 - 3*zb(&
      layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bb(layer)*at(layer)*cos(m(i)*zt(layer))*cos(m(&
      j)*zt(layer)) + 2*zb(layer)*m(j)*m(i)**3*1.0/(zb(layer)**2*m(i)**&
      6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)&
      **2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt&
      (layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4&
      *m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 -&
      3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 -&
      zt(layer)**2*m(j)**6)*bt(layer)*ab(layer)*cos(m(i)*zb(layer))*cos&
      (m(j)*zb(layer)) - 2*zb(layer)*m(j)*m(i)**3*1.0/(zb(layer)**2*m(i&
      )**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(&
      i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*&
      zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)&
      **4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**&
      6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)&
      **2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(layer)*cos(m(i)*zt(layer&
      ))*cos(m(j)*zt(layer)) - 4*zb(layer)*m(j)*m(i)**3*1.0/(zb(layer)&
      **2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j&
      )**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)&
      **6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(&
      layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer&
      )**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(&
      j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*at(layer)*cos(m(i&
      )*zb(layer))*cos(m(j)*zb(layer)) + 4*zb(layer)*m(j)*m(i)**3*1.0/(&
      zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(&
      layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb&
      (layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(&
      layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*at(&
      layer)*cos(m(i)*zt(layer))*cos(m(j)*zt(layer)) + zb(layer)*m(j)*m&
      (i)**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4&
      + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*&
      at(layer)*zb(layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) - zb(&
      layer)*m(j)*m(i)**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(&
      j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j&
      )**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)&
      **2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)&
      *zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**&
      2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6&
      )*bb(layer)*at(layer)*zt(layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(&
      layer)) + zb(layer)*m(j)*m(i)**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb&
      (layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bt(layer)*ab(layer)*zb(layer)*cos(m(j)*zb(&
      layer))*sin(m(i)*zb(layer)) - zb(layer)*m(j)*m(i)**4*1.0/(zb(&
      layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)&
      **2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer&
      )*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*&
      zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(&
      layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)&
      **2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(layer)*&
      zt(layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer)) - 2*zb(layer)*m&
      (j)*m(i)**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(&
      i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2&
      *zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)&
      **4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt&
      (layer)*at(layer)*zb(layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer&
      )) + 2*zb(layer)*m(j)*m(i)**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(&
      layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bt(layer)*at(layer)*zt(layer)*cos(m(j)*zt(&
      layer))*sin(m(i)*zt(layer)) + 2*zb(layer)*m(j)**2*m(i)**3*1.0/(zb&
      (layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)&
      **2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer&
      )*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*&
      zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(&
      layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)&
      **2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*at(layer)*&
      zb(layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) - 2*zb(layer)*m&
      (j)**2*m(i)**3*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2&
      *m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6&
      - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m&
      (i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb&
      (layer)*at(layer)*zt(layer)*cos(m(i)*zt(layer))*sin(m(j)*zt(layer&
      )) + 2*zb(layer)*m(j)**2*m(i)**3*1.0/(zb(layer)**2*m(i)**6 - 3*zb&
      (layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bt(layer)*ab(layer)*zb(layer)*cos(m(i)*zb(&
      layer))*sin(m(j)*zb(layer)) - 2*zb(layer)*m(j)**2*m(i)**3*1.0/(zb&
      (layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)&
      **2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer&
      )*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*&
      zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(&
      layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)&
      **2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(layer)*&
      zt(layer)*cos(m(i)*zt(layer))*sin(m(j)*zt(layer)) - 4*zb(layer)*m&
      (j)**2*m(i)**3*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2&
      *m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6&
      - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m&
      (i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt&
      (layer)*at(layer)*zb(layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer&
      )) + 4*zb(layer)*m(j)**2*m(i)**3*1.0/(zb(layer)**2*m(i)**6 - 3*zb&
      (layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bt(layer)*at(layer)*zt(layer)*cos(m(i)*zt(&
      layer))*sin(m(j)*zt(layer)) - 2*zb(layer)*m(j)**3*m(i)*1.0/(zb(&
      layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)&
      **2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer&
      )*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*&
      zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(&
      layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)&
      **2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*at(layer)*&
      cos(m(i)*zb(layer))*cos(m(j)*zb(layer)) + 2*zb(layer)*m(j)**3*m(i&
      )*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*&
      zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)&
      *zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt&
      (layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*at(&
      layer)*cos(m(i)*zt(layer))*cos(m(j)*zt(layer)) - 2*zb(layer)*m(j)&
      **3*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)&
      **4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*&
      zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)&
      **4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt&
      (layer)*ab(layer)*cos(m(i)*zb(layer))*cos(m(j)*zb(layer)) + 2*zb(&
      layer)*m(j)**3*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(&
      j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j&
      )**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)&
      **2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)&
      *zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**&
      2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6&
      )*bt(layer)*ab(layer)*cos(m(i)*zt(layer))*cos(m(j)*zt(layer)) + 4&
      *zb(layer)*m(j)**3*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**&
      2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2&
      *m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*&
      m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(&
      layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*&
      m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m&
      (j)**6)*bt(layer)*at(layer)*cos(m(i)*zb(layer))*cos(m(j)*zb(layer&
      )) - 4*zb(layer)*m(j)**3*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(&
      layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bt(layer)*at(layer)*cos(m(i)*zt(layer))*cos(m(&
      j)*zt(layer)) - 2*zb(layer)*m(j)**3*m(i)**2*1.0/(zb(layer)**2*m(i&
      )**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(&
      i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*&
      zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)&
      **4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**&
      6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)&
      **2 - zt(layer)**2*m(j)**6)*bb(layer)*at(layer)*zb(layer)*cos(m(j&
      )*zb(layer))*sin(m(i)*zb(layer)) + 2*zb(layer)*m(j)**3*m(i)**2*&
      1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb&
      (layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*&
      zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(&
      layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*at(&
      layer)*zt(layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer)) - 2*zb(&
      layer)*m(j)**3*m(i)**2*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2&
      *m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*&
      m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m&
      (j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(&
      layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*&
      m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m&
      (j)**6)*bt(layer)*ab(layer)*zb(layer)*cos(m(j)*zb(layer))*sin(m(i&
      )*zb(layer)) + 2*zb(layer)*m(j)**3*m(i)**2*1.0/(zb(layer)**2*m(i)&
      **6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i&
      )**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*&
      zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)&
      **4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**&
      6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)&
      **2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(layer)*zt(layer)*cos(m(j&
      )*zt(layer))*sin(m(i)*zt(layer)) + 4*zb(layer)*m(j)**3*m(i)**2*&
      1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb&
      (layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*&
      zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(&
      layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*at(&
      layer)*zb(layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) - 4*zb(&
      layer)*m(j)**3*m(i)**2*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2&
      *m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*&
      m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m&
      (j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(&
      layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*&
      m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m&
      (j)**6)*bt(layer)*at(layer)*zt(layer)*cos(m(j)*zt(layer))*sin(m(i&
      )*zt(layer)) - zb(layer)*m(j)**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb&
      (layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bb(layer)*at(layer)*sin(m(i)*zb(layer))*sin(m(&
      j)*zb(layer)) + zb(layer)*m(j)**4*1.0/(zb(layer)**2*m(i)**6 - 3*&
      zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 -&
      zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer&
      )*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)&
      **2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt&
      (layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bb(layer)*at(layer)*sin(m(i)*zt(layer))*sin(m(&
      j)*zt(layer)) - zb(layer)*m(j)**4*1.0/(zb(layer)**2*m(i)**6 - 3*&
      zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 -&
      zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer&
      )*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)&
      **2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt&
      (layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bt(layer)*ab(layer)*sin(m(i)*zb(layer))*sin(m(&
      j)*zb(layer)) + zb(layer)*m(j)**4*1.0/(zb(layer)**2*m(i)**6 - 3*&
      zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 -&
      zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer&
      )*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)&
      **2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt&
      (layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bt(layer)*ab(layer)*sin(m(i)*zt(layer))*sin(m(&
      j)*zt(layer)) + 2*zb(layer)*m(j)**4*1.0/(zb(layer)**2*m(i)**6 - 3&
      *zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 -&
      zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer&
      )*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)&
      **2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt&
      (layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bt(layer)*at(layer)*sin(m(i)*zb(layer))*sin(m(&
      j)*zb(layer)) - 2*zb(layer)*m(j)**4*1.0/(zb(layer)**2*m(i)**6 - 3&
      *zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 -&
      zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer&
      )*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)&
      **2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt&
      (layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bt(layer)*at(layer)*sin(m(i)*zt(layer))*sin(m(&
      j)*zt(layer)) - zb(layer)*m(j)**4*m(i)*1.0/(zb(layer)**2*m(i)**6&
      - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2&
      - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(&
      layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*&
      m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 -&
      3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 -&
      zt(layer)**2*m(j)**6)*bb(layer)*at(layer)*zb(layer)*cos(m(i)*zb(&
      layer))*sin(m(j)*zb(layer)) + zb(layer)*m(j)**4*m(i)*1.0/(zb(&
      layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)&
      **2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer&
      )*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*&
      zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(&
      layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)&
      **2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*at(layer)*&
      zt(layer)*cos(m(i)*zt(layer))*sin(m(j)*zt(layer)) - zb(layer)*m(j&
      )**4*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)&
      **4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*&
      zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)&
      **4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt&
      (layer)*ab(layer)*zb(layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer&
      )) + zb(layer)*m(j)**4*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(&
      layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bt(layer)*ab(layer)*zt(layer)*cos(m(i)*zt(&
      layer))*sin(m(j)*zt(layer)) + 2*zb(layer)*m(j)**4*m(i)*1.0/(zb(&
      layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)&
      **2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer&
      )*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*&
      zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(&
      layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)&
      **2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*at(layer)*&
      zb(layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) - 2*zb(layer)*m&
      (j)**4*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(&
      i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2&
      *zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)&
      **4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt&
      (layer)*at(layer)*zt(layer)*cos(m(i)*zt(layer))*sin(m(j)*zt(layer&
      )) + zb(layer)*m(j)**5*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2&
      *m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*&
      m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m&
      (j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(&
      layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*&
      m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m&
      (j)**6)*bb(layer)*at(layer)*zb(layer)*cos(m(j)*zb(layer))*sin(m(i&
      )*zb(layer)) - zb(layer)*m(j)**5*1.0/(zb(layer)**2*m(i)**6 - 3*zb&
      (layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bb(layer)*at(layer)*zt(layer)*cos(m(j)*zt(&
      layer))*sin(m(i)*zt(layer)) + zb(layer)*m(j)**5*1.0/(zb(layer)**2&
      *m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**&
      4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6&
      + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m&
      (j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i&
      )**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(&
      i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(layer)*zb(layer)*cos(m&
      (j)*zb(layer))*sin(m(i)*zb(layer)) - zb(layer)*m(j)**5*1.0/(zb(&
      layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)&
      **2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer&
      )*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*&
      zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(&
      layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)&
      **2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(layer)*&
      zt(layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer)) - 2*zb(layer)*m&
      (j)**5*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4&
      + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*&
      at(layer)*zb(layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) + 2*&
      zb(layer)*m(j)**5*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)&
      **2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)&
      **6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)&
      **2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)&
      *zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**&
      2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6&
      )*bt(layer)*at(layer)*zt(layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(&
      layer)) - zb(layer)**2*m(i)**5*1.0/(zb(layer)**2*m(i)**6 - 3*zb(&
      layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bt(layer)*at(layer)*cos(m(i)*zb(layer))*sin(m(&
      j)*zb(layer)) + zb(layer)**2*m(i)**5*1.0/(zb(layer)**2*m(i)**6 -&
      3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 -&
      zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer&
      )*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)&
      **2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt&
      (layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bt(layer)*at(layer)*cos(m(i)*zt(layer))*sin(m(&
      j)*zt(layer)) + zb(layer)**2*m(j)*m(i)**4*1.0/(zb(layer)**2*m(i)&
      **6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i&
      )**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*&
      zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)&
      **4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**&
      6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)&
      **2 - zt(layer)**2*m(j)**6)*bt(layer)*at(layer)*cos(m(j)*zb(layer&
      ))*sin(m(i)*zb(layer)) - zb(layer)**2*m(j)*m(i)**4*1.0/(zb(layer)&
      **2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j&
      )**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)&
      **6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(&
      layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer&
      )**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(&
      j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*at(layer)*cos(m(j&
      )*zt(layer))*sin(m(i)*zt(layer)) + 2*zb(layer)**2*m(j)**2*m(i)**3&
      *1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*&
      zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)&
      *zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt&
      (layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*at(&
      layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) - 2*zb(layer)**2*m&
      (j)**2*m(i)**3*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2&
      *m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6&
      - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m&
      (i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt&
      (layer)*at(layer)*cos(m(i)*zt(layer))*sin(m(j)*zt(layer)) - 2*zb(&
      layer)**2*m(j)**3*m(i)**2*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)&
      **2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)&
      **2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(&
      layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 +&
      2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer&
      )**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)&
      **2*m(j)**6)*bt(layer)*at(layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(&
      layer)) + 2*zb(layer)**2*m(j)**3*m(i)**2*1.0/(zb(layer)**2*m(i)**&
      6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)&
      **2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt&
      (layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4&
      *m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 -&
      3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 -&
      zt(layer)**2*m(j)**6)*bt(layer)*at(layer)*cos(m(j)*zt(layer))*sin&
      (m(i)*zt(layer)) - zb(layer)**2*m(j)**4*m(i)*1.0/(zb(layer)**2*m(&
      i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m&
      (i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6&
      *zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)&
      **4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**&
      6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)&
      **2 - zt(layer)**2*m(j)**6)*bt(layer)*at(layer)*cos(m(i)*zb(layer&
      ))*sin(m(j)*zb(layer)) + zb(layer)**2*m(j)**4*m(i)*1.0/(zb(layer)&
      **2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j&
      )**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)&
      **6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(&
      layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer&
      )**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(&
      j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*at(layer)*cos(m(i&
      )*zt(layer))*sin(m(j)*zt(layer)) + zb(layer)**2*m(j)**5*1.0/(zb(&
      layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)&
      **2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer&
      )*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*&
      zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(&
      layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)&
      **2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*at(layer)*&
      cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) - zb(layer)**2*m(j)**5*&
      1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb&
      (layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*&
      zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(&
      layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*at(&
      layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer)) - 2*zt(layer)*m(i)&
      **4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 +&
      3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*&
      ab(layer)*sin(m(i)*zb(layer))*sin(m(j)*zb(layer)) + 2*zt(layer)*m&
      (i)**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4&
      + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*&
      ab(layer)*sin(m(i)*zt(layer))*sin(m(j)*zt(layer)) + zt(layer)*m(i&
      )**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 +&
      3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*&
      at(layer)*sin(m(i)*zb(layer))*sin(m(j)*zb(layer)) - zt(layer)*m(i&
      )**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 +&
      3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*&
      at(layer)*sin(m(i)*zt(layer))*sin(m(j)*zt(layer)) + zt(layer)*m(i&
      )**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 +&
      3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*&
      ab(layer)*sin(m(i)*zb(layer))*sin(m(j)*zb(layer)) - zt(layer)*m(i&
      )**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 +&
      3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*&
      ab(layer)*sin(m(i)*zt(layer))*sin(m(j)*zt(layer)) + 2*zt(layer)*m&
      (i)**5*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4&
      + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*&
      ab(layer)*zb(layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) - 2*&
      zt(layer)*m(i)**5*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)&
      **2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)&
      **6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)&
      **2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)&
      *zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**&
      2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6&
      )*bb(layer)*ab(layer)*zt(layer)*cos(m(i)*zt(layer))*sin(m(j)*zt(&
      layer)) - zt(layer)*m(i)**5*1.0/(zb(layer)**2*m(i)**6 - 3*zb(&
      layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bb(layer)*at(layer)*zb(layer)*cos(m(i)*zb(&
      layer))*sin(m(j)*zb(layer)) + zt(layer)*m(i)**5*1.0/(zb(layer)**2&
      *m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**&
      4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6&
      + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m&
      (j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i&
      )**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(&
      i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*at(layer)*zt(layer)*cos(m&
      (i)*zt(layer))*sin(m(j)*zt(layer)) - zt(layer)*m(i)**5*1.0/(zb(&
      layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)&
      **2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer&
      )*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*&
      zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(&
      layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)&
      **2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(layer)*&
      zb(layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) + zt(layer)*m(i&
      )**5*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 +&
      3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*&
      ab(layer)*zt(layer)*cos(m(i)*zt(layer))*sin(m(j)*zt(layer)) - 4*&
      zt(layer)*m(j)*m(i)**3*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2&
      *m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*&
      m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m&
      (j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(&
      layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*&
      m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m&
      (j)**6)*bb(layer)*ab(layer)*cos(m(i)*zb(layer))*cos(m(j)*zb(layer&
      )) + 4*zt(layer)*m(j)*m(i)**3*1.0/(zb(layer)**2*m(i)**6 - 3*zb(&
      layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bb(layer)*ab(layer)*cos(m(i)*zt(layer))*cos(m(&
      j)*zt(layer)) + 2*zt(layer)*m(j)*m(i)**3*1.0/(zb(layer)**2*m(i)**&
      6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)&
      **2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt&
      (layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4&
      *m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 -&
      3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 -&
      zt(layer)**2*m(j)**6)*bb(layer)*at(layer)*cos(m(i)*zb(layer))*cos&
      (m(j)*zb(layer)) - 2*zt(layer)*m(j)*m(i)**3*1.0/(zb(layer)**2*m(i&
      )**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(&
      i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*&
      zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)&
      **4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**&
      6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)&
      **2 - zt(layer)**2*m(j)**6)*bb(layer)*at(layer)*cos(m(i)*zt(layer&
      ))*cos(m(j)*zt(layer)) + 2*zt(layer)*m(j)*m(i)**3*1.0/(zb(layer)&
      **2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j&
      )**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)&
      **6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(&
      layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer&
      )**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(&
      j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(layer)*cos(m(i&
      )*zb(layer))*cos(m(j)*zb(layer)) - 2*zt(layer)*m(j)*m(i)**3*1.0/(&
      zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(&
      layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb&
      (layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(&
      layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(&
      layer)*cos(m(i)*zt(layer))*cos(m(j)*zt(layer)) - 2*zt(layer)*m(j)&
      *m(i)**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)&
      **4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*&
      zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)&
      **4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb&
      (layer)*ab(layer)*zb(layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer&
      )) + 2*zt(layer)*m(j)*m(i)**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(&
      layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bb(layer)*ab(layer)*zt(layer)*cos(m(j)*zt(&
      layer))*sin(m(i)*zt(layer)) + zt(layer)*m(j)*m(i)**4*1.0/(zb(&
      layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)&
      **2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer&
      )*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*&
      zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(&
      layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)&
      **2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*at(layer)*&
      zb(layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) - zt(layer)*m(j&
      )*m(i)**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)&
      **4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*&
      zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)&
      **4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb&
      (layer)*at(layer)*zt(layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer&
      )) + zt(layer)*m(j)*m(i)**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(&
      layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bt(layer)*ab(layer)*zb(layer)*cos(m(j)*zb(&
      layer))*sin(m(i)*zb(layer)) - zt(layer)*m(j)*m(i)**4*1.0/(zb(&
      layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)&
      **2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer&
      )*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*&
      zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(&
      layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)&
      **2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(layer)*&
      zt(layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer)) - 4*zt(layer)*m&
      (j)**2*m(i)**3*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2&
      *m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6&
      - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m&
      (i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb&
      (layer)*ab(layer)*zb(layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer&
      )) + 4*zt(layer)*m(j)**2*m(i)**3*1.0/(zb(layer)**2*m(i)**6 - 3*zb&
      (layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bb(layer)*ab(layer)*zt(layer)*cos(m(i)*zt(&
      layer))*sin(m(j)*zt(layer)) + 2*zt(layer)*m(j)**2*m(i)**3*1.0/(zb&
      (layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)&
      **2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer&
      )*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*&
      zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(&
      layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)&
      **2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*at(layer)*&
      zb(layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) - 2*zt(layer)*m&
      (j)**2*m(i)**3*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2&
      *m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6&
      - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m&
      (i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb&
      (layer)*at(layer)*zt(layer)*cos(m(i)*zt(layer))*sin(m(j)*zt(layer&
      )) + 2*zt(layer)*m(j)**2*m(i)**3*1.0/(zb(layer)**2*m(i)**6 - 3*zb&
      (layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bt(layer)*ab(layer)*zb(layer)*cos(m(i)*zb(&
      layer))*sin(m(j)*zb(layer)) - 2*zt(layer)*m(j)**2*m(i)**3*1.0/(zb&
      (layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)&
      **2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer&
      )*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*&
      zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(&
      layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)&
      **2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(layer)*&
      zt(layer)*cos(m(i)*zt(layer))*sin(m(j)*zt(layer)) + 4*zt(layer)*m&
      (j)**3*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(&
      i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2&
      *zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)&
      **4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb&
      (layer)*ab(layer)*cos(m(i)*zb(layer))*cos(m(j)*zb(layer)) - 4*zt(&
      layer)*m(j)**3*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(&
      j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j&
      )**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)&
      **2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)&
      *zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**&
      2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6&
      )*bb(layer)*ab(layer)*cos(m(i)*zt(layer))*cos(m(j)*zt(layer)) - 2&
      *zt(layer)*m(j)**3*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**&
      2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2&
      *m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*&
      m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(&
      layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*&
      m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m&
      (j)**6)*bb(layer)*at(layer)*cos(m(i)*zb(layer))*cos(m(j)*zb(layer&
      )) + 2*zt(layer)*m(j)**3*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(&
      layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bb(layer)*at(layer)*cos(m(i)*zt(layer))*cos(m(&
      j)*zt(layer)) - 2*zt(layer)*m(j)**3*m(i)*1.0/(zb(layer)**2*m(i)**&
      6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)&
      **2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt&
      (layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4&
      *m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 -&
      3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 -&
      zt(layer)**2*m(j)**6)*bt(layer)*ab(layer)*cos(m(i)*zb(layer))*cos&
      (m(j)*zb(layer)) + 2*zt(layer)*m(j)**3*m(i)*1.0/(zb(layer)**2*m(i&
      )**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(&
      i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*&
      zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)&
      **4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**&
      6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)&
      **2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(layer)*cos(m(i)*zt(layer&
      ))*cos(m(j)*zt(layer)) + 4*zt(layer)*m(j)**3*m(i)**2*1.0/(zb(&
      layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)&
      **2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer&
      )*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*&
      zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(&
      layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)&
      **2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*ab(layer)*&
      zb(layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) - 4*zt(layer)*m&
      (j)**3*m(i)**2*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2&
      *m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6&
      - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m&
      (i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb&
      (layer)*ab(layer)*zt(layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer&
      )) - 2*zt(layer)*m(j)**3*m(i)**2*1.0/(zb(layer)**2*m(i)**6 - 3*zb&
      (layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bb(layer)*at(layer)*zb(layer)*cos(m(j)*zb(&
      layer))*sin(m(i)*zb(layer)) + 2*zt(layer)*m(j)**3*m(i)**2*1.0/(zb&
      (layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)&
      **2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer&
      )*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*&
      zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(&
      layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)&
      **2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*at(layer)*&
      zt(layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer)) - 2*zt(layer)*m&
      (j)**3*m(i)**2*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2&
      *m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6&
      - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m&
      (i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt&
      (layer)*ab(layer)*zb(layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer&
      )) + 2*zt(layer)*m(j)**3*m(i)**2*1.0/(zb(layer)**2*m(i)**6 - 3*zb&
      (layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bt(layer)*ab(layer)*zt(layer)*cos(m(j)*zt(&
      layer))*sin(m(i)*zt(layer)) + 2*zt(layer)*m(j)**4*1.0/(zb(layer)&
      **2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j&
      )**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)&
      **6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(&
      layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer&
      )**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(&
      j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*ab(layer)*sin(m(i&
      )*zb(layer))*sin(m(j)*zb(layer)) - 2*zt(layer)*m(j)**4*1.0/(zb(&
      layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)&
      **2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer&
      )*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*&
      zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(&
      layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)&
      **2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*ab(layer)*&
      sin(m(i)*zt(layer))*sin(m(j)*zt(layer)) - zt(layer)*m(j)**4*1.0/(&
      zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(&
      layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb&
      (layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(&
      layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*at(&
      layer)*sin(m(i)*zb(layer))*sin(m(j)*zb(layer)) + zt(layer)*m(j)**&
      4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*&
      zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)&
      *zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt&
      (layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*at(&
      layer)*sin(m(i)*zt(layer))*sin(m(j)*zt(layer)) - zt(layer)*m(j)**&
      4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*&
      zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)&
      *zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt&
      (layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(&
      layer)*sin(m(i)*zb(layer))*sin(m(j)*zb(layer)) + zt(layer)*m(j)**&
      4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*&
      zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)&
      *zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt&
      (layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(&
      layer)*sin(m(i)*zt(layer))*sin(m(j)*zt(layer)) + 2*zt(layer)*m(j)&
      **4*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)&
      **4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*&
      zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)&
      **4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb&
      (layer)*ab(layer)*zb(layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer&
      )) - 2*zt(layer)*m(j)**4*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(&
      layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bb(layer)*ab(layer)*zt(layer)*cos(m(i)*zt(&
      layer))*sin(m(j)*zt(layer)) - zt(layer)*m(j)**4*m(i)*1.0/(zb(&
      layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)&
      **2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer&
      )*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*&
      zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(&
      layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)&
      **2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*at(layer)*&
      zb(layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) + zt(layer)*m(j&
      )**4*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)&
      **4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*&
      zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)&
      **4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb&
      (layer)*at(layer)*zt(layer)*cos(m(i)*zt(layer))*sin(m(j)*zt(layer&
      )) - zt(layer)*m(j)**4*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(&
      layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bt(layer)*ab(layer)*zb(layer)*cos(m(i)*zb(&
      layer))*sin(m(j)*zb(layer)) + zt(layer)*m(j)**4*m(i)*1.0/(zb(&
      layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)&
      **2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer&
      )*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*&
      zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(&
      layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)&
      **2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(layer)*&
      zt(layer)*cos(m(i)*zt(layer))*sin(m(j)*zt(layer)) - 2*zt(layer)*m&
      (j)**5*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4&
      + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*&
      ab(layer)*zb(layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) + 2*&
      zt(layer)*m(j)**5*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)&
      **2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)&
      **6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)&
      **2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)&
      *zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**&
      2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6&
      )*bb(layer)*ab(layer)*zt(layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(&
      layer)) + zt(layer)*m(j)**5*1.0/(zb(layer)**2*m(i)**6 - 3*zb(&
      layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bb(layer)*at(layer)*zb(layer)*cos(m(j)*zb(&
      layer))*sin(m(i)*zb(layer)) - zt(layer)*m(j)**5*1.0/(zb(layer)**2&
      *m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**&
      4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6&
      + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m&
      (j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i&
      )**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(&
      i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*at(layer)*zt(layer)*cos(m&
      (j)*zt(layer))*sin(m(i)*zt(layer)) + zt(layer)*m(j)**5*1.0/(zb(&
      layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)&
      **2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer&
      )*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*&
      zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(&
      layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)&
      **2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(layer)*&
      zb(layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) - zt(layer)*m(j&
      )**5*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 +&
      3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*&
      ab(layer)*zt(layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer)) + zt(&
      layer)*zb(layer)*m(i)**5*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)&
      **2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)&
      **2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(&
      layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 +&
      2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer&
      )**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)&
      **2*m(j)**6)*bb(layer)*at(layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(&
      layer)) - zt(layer)*zb(layer)*m(i)**5*1.0/(zb(layer)**2*m(i)**6 -&
      3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 -&
      zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer&
      )*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)&
      **2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt&
      (layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bb(layer)*at(layer)*cos(m(i)*zt(layer))*sin(m(&
      j)*zt(layer)) + zt(layer)*zb(layer)*m(i)**5*1.0/(zb(layer)**2*m(i&
      )**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(&
      i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*&
      zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)&
      **4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**&
      6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)&
      **2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(layer)*cos(m(i)*zb(layer&
      ))*sin(m(j)*zb(layer)) - zt(layer)*zb(layer)*m(i)**5*1.0/(zb(&
      layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)&
      **2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer&
      )*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*&
      zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(&
      layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)&
      **2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(layer)*&
      cos(m(i)*zt(layer))*sin(m(j)*zt(layer)) - zt(layer)*zb(layer)*m(j&
      )*m(i)**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)&
      **4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*&
      zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)&
      **4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb&
      (layer)*at(layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) + zt(&
      layer)*zb(layer)*m(j)*m(i)**4*1.0/(zb(layer)**2*m(i)**6 - 3*zb(&
      layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bb(layer)*at(layer)*cos(m(j)*zt(layer))*sin(m(&
      i)*zt(layer)) - zt(layer)*zb(layer)*m(j)*m(i)**4*1.0/(zb(layer)**&
      2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)&
      **4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**&
      6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)&
      *m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m&
      (i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*&
      m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(layer)*cos(m(j)*zb(&
      layer))*sin(m(i)*zb(layer)) + zt(layer)*zb(layer)*m(j)*m(i)**4*&
      1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb&
      (layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*&
      zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(&
      layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(&
      layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer)) - 2*zt(layer)*zb(&
      layer)*m(j)**2*m(i)**3*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2&
      *m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*&
      m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m&
      (j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(&
      layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*&
      m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m&
      (j)**6)*bb(layer)*at(layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer&
      )) + 2*zt(layer)*zb(layer)*m(j)**2*m(i)**3*1.0/(zb(layer)**2*m(i)&
      **6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i&
      )**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*&
      zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)&
      **4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**&
      6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)&
      **2 - zt(layer)**2*m(j)**6)*bb(layer)*at(layer)*cos(m(i)*zt(layer&
      ))*sin(m(j)*zt(layer)) - 2*zt(layer)*zb(layer)*m(j)**2*m(i)**3*&
      1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb&
      (layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*&
      zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(&
      layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(&
      layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) + 2*zt(layer)*zb(&
      layer)*m(j)**2*m(i)**3*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2&
      *m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*&
      m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m&
      (j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(&
      layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*&
      m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m&
      (j)**6)*bt(layer)*ab(layer)*cos(m(i)*zt(layer))*sin(m(j)*zt(layer&
      )) + 2*zt(layer)*zb(layer)*m(j)**3*m(i)**2*1.0/(zb(layer)**2*m(i)&
      **6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i&
      )**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*&
      zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)&
      **4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**&
      6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)&
      **2 - zt(layer)**2*m(j)**6)*bb(layer)*at(layer)*cos(m(j)*zb(layer&
      ))*sin(m(i)*zb(layer)) - 2*zt(layer)*zb(layer)*m(j)**3*m(i)**2*&
      1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb&
      (layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*&
      zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(&
      layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*at(&
      layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer)) + 2*zt(layer)*zb(&
      layer)*m(j)**3*m(i)**2*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2&
      *m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*&
      m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m&
      (j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(&
      layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*&
      m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m&
      (j)**6)*bt(layer)*ab(layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer&
      )) - 2*zt(layer)*zb(layer)*m(j)**3*m(i)**2*1.0/(zb(layer)**2*m(i)&
      **6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i&
      )**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*&
      zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)&
      **4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**&
      6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)&
      **2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(layer)*cos(m(j)*zt(layer&
      ))*sin(m(i)*zt(layer)) + zt(layer)*zb(layer)*m(j)**4*m(i)*1.0/(zb&
      (layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)&
      **2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer&
      )*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*&
      zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(&
      layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)&
      **2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*at(layer)*&
      cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) - zt(layer)*zb(layer)*m(j&
      )**4*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)&
      **4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*&
      zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)&
      **4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb&
      (layer)*at(layer)*cos(m(i)*zt(layer))*sin(m(j)*zt(layer)) + zt(&
      layer)*zb(layer)*m(j)**4*m(i)*1.0/(zb(layer)**2*m(i)**6 - 3*zb(&
      layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bt(layer)*ab(layer)*cos(m(i)*zb(layer))*sin(m(&
      j)*zb(layer)) - zt(layer)*zb(layer)*m(j)**4*m(i)*1.0/(zb(layer)**&
      2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)&
      **4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**&
      6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)&
      *m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m&
      (i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*&
      m(i)**2 - zt(layer)**2*m(j)**6)*bt(layer)*ab(layer)*cos(m(i)*zt(&
      layer))*sin(m(j)*zt(layer)) - zt(layer)*zb(layer)*m(j)**5*1.0/(zb&
      (layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)&
      **2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer&
      )*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*&
      zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(&
      layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)&
      **2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*at(layer)*&
      cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) + zt(layer)*zb(layer)*m(j&
      )**5*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 +&
      3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(&
      layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4&
      - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m&
      (j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 +&
      3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*&
      at(layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer)) - zt(layer)*zb(&
      layer)*m(j)**5*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2&
      *m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6&
      - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m&
      (i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bt&
      (layer)*ab(layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) + zt(&
      layer)*zb(layer)*m(j)**5*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)&
      **2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)&
      **2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(&
      layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 +&
      2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer&
      )**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)&
      **2*m(j)**6)*bt(layer)*ab(layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(&
      layer)) - zt(layer)**2*m(i)**5*1.0/(zb(layer)**2*m(i)**6 - 3*zb(&
      layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(&
      layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*&
      zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2&
      + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(&
      layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bb(layer)*ab(layer)*cos(m(i)*zb(layer))*sin(m(&
      j)*zb(layer)) + zt(layer)**2*m(i)**5*1.0/(zb(layer)**2*m(i)**6 -&
      3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 -&
      zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer&
      )*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)&
      **2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt&
      (layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(&
      layer)**2*m(j)**6)*bb(layer)*ab(layer)*cos(m(i)*zt(layer))*sin(m(&
      j)*zt(layer)) + zt(layer)**2*m(j)*m(i)**4*1.0/(zb(layer)**2*m(i)&
      **6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i&
      )**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*&
      zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)&
      **4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**&
      6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)&
      **2 - zt(layer)**2*m(j)**6)*bb(layer)*ab(layer)*cos(m(j)*zb(layer&
      ))*sin(m(i)*zb(layer)) - zt(layer)**2*m(j)*m(i)**4*1.0/(zb(layer)&
      **2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j&
      )**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)&
      **6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(&
      layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer&
      )**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(&
      j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*ab(layer)*cos(m(j&
      )*zt(layer))*sin(m(i)*zt(layer)) + 2*zt(layer)**2*m(j)**2*m(i)**3&
      *1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*&
      zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)&
      *zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt&
      (layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*ab(&
      layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) - 2*zt(layer)**2*m&
      (j)**2*m(i)**3*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2&
      *m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6&
      - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m&
      (i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(&
      layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(&
      i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb&
      (layer)*ab(layer)*cos(m(i)*zt(layer))*sin(m(j)*zt(layer)) - 2*zt(&
      layer)**2*m(j)**3*m(i)**2*1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)&
      **2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)**2 - zb(layer)&
      **2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt(layer)*zb(&
      layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4*m(i)**2 +&
      2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 - 3*zt(layer&
      )**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 - zt(layer)&
      **2*m(j)**6)*bb(layer)*ab(layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(&
      layer)) + 2*zt(layer)**2*m(j)**3*m(i)**2*1.0/(zb(layer)**2*m(i)**&
      6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m(i)&
      **2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6*zt&
      (layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)**4&
      *m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**6 -&
      3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)**2 -&
      zt(layer)**2*m(j)**6)*bb(layer)*ab(layer)*cos(m(j)*zt(layer))*sin&
      (m(i)*zt(layer)) - zt(layer)**2*m(j)**4*m(i)*1.0/(zb(layer)**2*m(&
      i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j)**4*m&
      (i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)**6 + 6&
      *zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(layer)*m(j)&
      **4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer)**2*m(i)**&
      6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(j)**4*m(i)&
      **2 - zt(layer)**2*m(j)**6)*bb(layer)*ab(layer)*cos(m(i)*zb(layer&
      ))*sin(m(j)*zb(layer)) + zt(layer)**2*m(j)**4*m(i)*1.0/(zb(layer)&
      **2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)**2*m(j&
      )**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer)*m(i)&
      **6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*zb(&
      layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(layer&
      )**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)**2*m(&
      j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*ab(layer)*cos(m(i&
      )*zt(layer))*sin(m(j)*zt(layer)) + zt(layer)**2*m(j)**5*1.0/(zb(&
      layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb(layer)&
      **2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*zb(layer&
      )*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(layer)*&
      zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6 + zt(&
      layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(layer)&
      **2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*ab(layer)*&
      cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) - zt(layer)**2*m(j)**5*&
      1.0/(zb(layer)**2*m(i)**6 - 3*zb(layer)**2*m(j)**2*m(i)**4 + 3*zb&
      (layer)**2*m(j)**4*m(i)**2 - zb(layer)**2*m(j)**6 - 2*zt(layer)*&
      zb(layer)*m(i)**6 + 6*zt(layer)*zb(layer)*m(j)**2*m(i)**4 - 6*zt(&
      layer)*zb(layer)*m(j)**4*m(i)**2 + 2*zt(layer)*zb(layer)*m(j)**6&
      + zt(layer)**2*m(i)**6 - 3*zt(layer)**2*m(j)**2*m(i)**4 + 3*zt(&
      layer)**2*m(j)**4*m(i)**2 - zt(layer)**2*m(j)**6)*bb(layer)*ab(&
      layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer)))
            END DO
          END DO
        END DO

        DO j = 0, neig -2
          DO i = j + 1, neig-1
            a(j,i) = a(i, j)
          END DO
        END DO
                    
      END SUBROUTINE

      SUBROUTINE dim1sin_D_aDf_linear(m, at, ab, zt, zb, a, neig, &
                                      nlayers) 
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
        a=0.0D0
        DO layer = 0, nlayers-1
          DO j = 0, neig-1
              i=j
      a(i, i) = a(i, i) + (1.0/(zb(layer) - zt(layer))*(ab(layer) - at(&
      layer))*sin(m(i)*zb(layer))**2/2 - 1.0/(zb(layer) - zt(layer))*(&
      ab(layer) - at(layer))*sin(m(i)*zt(layer))**2/2 - m(i)*(1.0/(zb(&
      layer) - zt(layer))*(ab(layer) - at(layer))*(zb(layer) - zt(layer&
      )) + at(layer))*cos(m(i)*zb(layer))*sin(m(i)*zb(layer)) + m(i)*at&
      (layer)*cos(m(i)*zt(layer))*sin(m(i)*zt(layer)) - m(i)**2*(-1.0/(&
      4*zb(layer)*m(i)**2 - 4*zt(layer)*m(i)**2)*ab(layer)*cos(m(i)*zb(&
      layer))**2 + 1.0/(4*zb(layer)*m(i)**2 - 4*zt(layer)*m(i)**2)*at(&
      layer)*cos(m(i)*zb(layer))**2 - 2*m(i)*1.0/(4*zb(layer)*m(i)**2 -&
      4*zt(layer)*m(i)**2)*ab(layer)*zb(layer)*cos(m(i)*zb(layer))*sin(&
      m(i)*zb(layer)) + 2*m(i)*1.0/(4*zb(layer)*m(i)**2 - 4*zt(layer)*m&
      (i)**2)*at(layer)*zb(layer)*cos(m(i)*zb(layer))*sin(m(i)*zb(layer&
      )) + m(i)**2*1.0/(4*zb(layer)*m(i)**2 - 4*zt(layer)*m(i)**2)*ab(&
      layer)*zb(layer)**2*sin(m(i)*zb(layer))**2 + m(i)**2*1.0/(4*zb(&
      layer)*m(i)**2 - 4*zt(layer)*m(i)**2)*ab(layer)*zb(layer)**2*cos(&
      m(i)*zb(layer))**2 - m(i)**2*1.0/(4*zb(layer)*m(i)**2 - 4*zt(&
      layer)*m(i)**2)*at(layer)*zb(layer)**2*sin(m(i)*zb(layer))**2 - m&
      (i)**2*1.0/(4*zb(layer)*m(i)**2 - 4*zt(layer)*m(i)**2)*at(layer)*&
      zb(layer)**2*cos(m(i)*zb(layer))**2 - 2*zb(layer)*m(i)*1.0/(4*zb(&
      layer)*m(i)**2 - 4*zt(layer)*m(i)**2)*at(layer)*cos(m(i)*zb(layer&
      ))*sin(m(i)*zb(layer)) + 2*zb(layer)*m(i)**2*1.0/(4*zb(layer)*m(i&
      )**2 - 4*zt(layer)*m(i)**2)*at(layer)*zb(layer)*sin(m(i)*zb(layer&
      ))**2 + 2*zb(layer)*m(i)**2*1.0/(4*zb(layer)*m(i)**2 - 4*zt(layer&
      )*m(i)**2)*at(layer)*zb(layer)*cos(m(i)*zb(layer))**2 + 2*zt(&
      layer)*m(i)*1.0/(4*zb(layer)*m(i)**2 - 4*zt(layer)*m(i)**2)*ab(&
      layer)*cos(m(i)*zb(layer))*sin(m(i)*zb(layer)) - 2*zt(layer)*m(i)&
      **2*1.0/(4*zb(layer)*m(i)**2 - 4*zt(layer)*m(i)**2)*ab(layer)*zb(&
      layer)*sin(m(i)*zb(layer))**2 - 2*zt(layer)*m(i)**2*1.0/(4*zb(&
      layer)*m(i)**2 - 4*zt(layer)*m(i)**2)*ab(layer)*zb(layer)*cos(m(i&
      )*zb(layer))**2) + m(i)**2*(-1.0/(4*zb(layer)*m(i)**2 - 4*zt(&
      layer)*m(i)**2)*ab(layer)*cos(m(i)*zt(layer))**2 + 1.0/(4*zb(&
      layer)*m(i)**2 - 4*zt(layer)*m(i)**2)*at(layer)*cos(m(i)*zt(layer&
      ))**2 - 2*m(i)*1.0/(4*zb(layer)*m(i)**2 - 4*zt(layer)*m(i)**2)*ab&
      (layer)*zt(layer)*cos(m(i)*zt(layer))*sin(m(i)*zt(layer)) + 2*m(i&
      )*1.0/(4*zb(layer)*m(i)**2 - 4*zt(layer)*m(i)**2)*at(layer)*zt(&
      layer)*cos(m(i)*zt(layer))*sin(m(i)*zt(layer)) + m(i)**2*1.0/(4*&
      zb(layer)*m(i)**2 - 4*zt(layer)*m(i)**2)*ab(layer)*zt(layer)**2*&
      sin(m(i)*zt(layer))**2 + m(i)**2*1.0/(4*zb(layer)*m(i)**2 - 4*zt(&
      layer)*m(i)**2)*ab(layer)*zt(layer)**2*cos(m(i)*zt(layer))**2 - m&
      (i)**2*1.0/(4*zb(layer)*m(i)**2 - 4*zt(layer)*m(i)**2)*at(layer)*&
      zt(layer)**2*sin(m(i)*zt(layer))**2 - m(i)**2*1.0/(4*zb(layer)*m(&
      i)**2 - 4*zt(layer)*m(i)**2)*at(layer)*zt(layer)**2*cos(m(i)*zt(&
      layer))**2 - 2*zb(layer)*m(i)*1.0/(4*zb(layer)*m(i)**2 - 4*zt(&
      layer)*m(i)**2)*at(layer)*cos(m(i)*zt(layer))*sin(m(i)*zt(layer&
      )) + 2*zb(layer)*m(i)**2*1.0/(4*zb(layer)*m(i)**2 - 4*zt(layer)*m&
      (i)**2)*at(layer)*zt(layer)*sin(m(i)*zt(layer))**2 + 2*zb(layer)*&
      m(i)**2*1.0/(4*zb(layer)*m(i)**2 - 4*zt(layer)*m(i)**2)*at(layer)&
      *zt(layer)*cos(m(i)*zt(layer))**2 + 2*zt(layer)*m(i)*1.0/(4*zb(&
      layer)*m(i)**2 - 4*zt(layer)*m(i)**2)*ab(layer)*cos(m(i)*zt(layer&
      ))*sin(m(i)*zt(layer)) - 2*zt(layer)*m(i)**2*1.0/(4*zb(layer)*m(i&
      )**2 - 4*zt(layer)*m(i)**2)*ab(layer)*zt(layer)*sin(m(i)*zt(layer&
      ))**2 - 2*zt(layer)*m(i)**2*1.0/(4*zb(layer)*m(i)**2 - 4*zt(layer&
      )*m(i)**2)*ab(layer)*zt(layer)*cos(m(i)*zt(layer))**2))
            DO i = j+1, neig-1
      a(i, j) = a(i, j) + (1.0/(zb(layer) - zt(layer))*m(j)*(ab(layer) -&
      at(layer))*(-m(i)*1.0/(m(i)**2 - m(j)**2)*cos(m(i)*zb(layer))*cos&
      (m(j)*zb(layer)) - m(j)*1.0/(m(i)**2 - m(j)**2)*sin(m(i)*zb(layer&
      ))*sin(m(j)*zb(layer))) - 1.0/(zb(layer) - zt(layer))*m(j)*(ab(&
      layer) - at(layer))*(-m(i)*1.0/(m(i)**2 - m(j)**2)*cos(m(i)*zt(&
      layer))*cos(m(j)*zt(layer)) - m(j)*1.0/(m(i)**2 - m(j)**2)*sin(m(&
      i)*zt(layer))*sin(m(j)*zt(layer))) - m(j)*(1.0/(zb(layer) - zt(&
      layer))*(ab(layer) - at(layer))*(zb(layer) - zt(layer)) + at(&
      layer))*cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) + m(j)*at(layer)*&
      cos(m(j)*zt(layer))*sin(m(i)*zt(layer)) - m(j)**2*(m(i)**2*1.0/(&
      zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)&
      **4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)&
      *m(j)**4)*ab(layer)*sin(m(i)*zb(layer))*sin(m(j)*zb(layer)) - m(i&
      )**2*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(&
      layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2&
      - zt(layer)*m(j)**4)*at(layer)*sin(m(i)*zb(layer))*sin(m(j)*zb(&
      layer)) - m(i)**3*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(&
      i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)&
      **2*m(i)**2 - zt(layer)*m(j)**4)*ab(layer)*zb(layer)*cos(m(i)*zb(&
      layer))*sin(m(j)*zb(layer)) + m(i)**3*1.0/(zb(layer)*m(i)**4 - 2*&
      zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4&
      + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)*at(layer)*zb(&
      layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) + 2*m(j)*m(i)*1.0/&
      (zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)&
      **4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)&
      *m(j)**4)*ab(layer)*cos(m(i)*zb(layer))*cos(m(j)*zb(layer)) - 2*m&
      (j)*m(i)*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 +&
      zb(layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)&
      **2 - zt(layer)*m(j)**4)*at(layer)*cos(m(i)*zb(layer))*cos(m(j)*&
      zb(layer)) + m(j)*m(i)**2*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(&
      j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(&
      layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)*ab(layer)*zb(layer)*&
      cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) - m(j)*m(i)**2*1.0/(zb(&
      layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4&
      - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j&
      )**4)*at(layer)*zb(layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer&
      )) + m(j)**2*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2&
      + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i&
      )**2 - zt(layer)*m(j)**4)*ab(layer)*sin(m(i)*zb(layer))*sin(m(j)*&
      zb(layer)) - m(j)**2*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2&
      *m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(&
      j)**2*m(i)**2 - zt(layer)*m(j)**4)*at(layer)*sin(m(i)*zb(layer))*&
      sin(m(j)*zb(layer)) + m(j)**2*m(i)*1.0/(zb(layer)*m(i)**4 - 2*zb(&
      layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 +&
      2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)*ab(layer)*zb(&
      layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) - m(j)**2*m(i)*1.0&
      /(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j&
      )**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer&
      )*m(j)**4)*at(layer)*zb(layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(&
      layer)) - m(j)**3*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(&
      i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)&
      **2*m(i)**2 - zt(layer)*m(j)**4)*ab(layer)*zb(layer)*cos(m(j)*zb(&
      layer))*sin(m(i)*zb(layer)) + m(j)**3*1.0/(zb(layer)*m(i)**4 - 2*&
      zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4&
      + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)*at(layer)*zb(&
      layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) - zb(layer)*m(i)**&
      3*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(layer&
      )*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(&
      layer)*m(j)**4)*at(layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer&
      )) + zb(layer)*m(j)*m(i)**2*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*&
      m(j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(&
      layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)*at(layer)*cos(m(j)*zb&
      (layer))*sin(m(i)*zb(layer)) + zb(layer)*m(j)**2*m(i)*1.0/(zb(&
      layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4&
      - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j&
      )**4)*at(layer)*cos(m(i)*zb(layer))*sin(m(j)*zb(layer)) - zb(&
      layer)*m(j)**3*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)&
      **2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2&
      *m(i)**2 - zt(layer)*m(j)**4)*at(layer)*cos(m(j)*zb(layer))*sin(m&
      (i)*zb(layer)) + zt(layer)*m(i)**3*1.0/(zb(layer)*m(i)**4 - 2*zb(&
      layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 +&
      2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)*ab(layer)*cos(m(&
      i)*zb(layer))*sin(m(j)*zb(layer)) - zt(layer)*m(j)*m(i)**2*1.0/(&
      zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)&
      **4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)&
      *m(j)**4)*ab(layer)*cos(m(j)*zb(layer))*sin(m(i)*zb(layer)) - zt(&
      layer)*m(j)**2*m(i)*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*&
      m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j&
      )**2*m(i)**2 - zt(layer)*m(j)**4)*ab(layer)*cos(m(i)*zb(layer))*&
      sin(m(j)*zb(layer)) + zt(layer)*m(j)**3*1.0/(zb(layer)*m(i)**4 -&
      2*zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)&
      **4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)*ab(layer)*&
      cos(m(j)*zb(layer))*sin(m(i)*zb(layer))) + m(j)**2*(m(i)**2*1.0/(&
      zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)&
      **4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)&
      *m(j)**4)*ab(layer)*sin(m(i)*zt(layer))*sin(m(j)*zt(layer)) - m(i&
      )**2*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(&
      layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2&
      - zt(layer)*m(j)**4)*at(layer)*sin(m(i)*zt(layer))*sin(m(j)*zt(&
      layer)) - m(i)**3*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(&
      i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)&
      **2*m(i)**2 - zt(layer)*m(j)**4)*ab(layer)*zt(layer)*cos(m(i)*zt(&
      layer))*sin(m(j)*zt(layer)) + m(i)**3*1.0/(zb(layer)*m(i)**4 - 2*&
      zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4&
      + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)*at(layer)*zt(&
      layer)*cos(m(i)*zt(layer))*sin(m(j)*zt(layer)) + 2*m(j)*m(i)*1.0/&
      (zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)&
      **4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)&
      *m(j)**4)*ab(layer)*cos(m(i)*zt(layer))*cos(m(j)*zt(layer)) - 2*m&
      (j)*m(i)*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 +&
      zb(layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)&
      **2 - zt(layer)*m(j)**4)*at(layer)*cos(m(i)*zt(layer))*cos(m(j)*&
      zt(layer)) + m(j)*m(i)**2*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(&
      j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(&
      layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)*ab(layer)*zt(layer)*&
      cos(m(j)*zt(layer))*sin(m(i)*zt(layer)) - m(j)*m(i)**2*1.0/(zb(&
      layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4&
      - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j&
      )**4)*at(layer)*zt(layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer&
      )) + m(j)**2*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2&
      + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i&
      )**2 - zt(layer)*m(j)**4)*ab(layer)*sin(m(i)*zt(layer))*sin(m(j)*&
      zt(layer)) - m(j)**2*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2&
      *m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(&
      j)**2*m(i)**2 - zt(layer)*m(j)**4)*at(layer)*sin(m(i)*zt(layer))*&
      sin(m(j)*zt(layer)) + m(j)**2*m(i)*1.0/(zb(layer)*m(i)**4 - 2*zb(&
      layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 +&
      2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)*ab(layer)*zt(&
      layer)*cos(m(i)*zt(layer))*sin(m(j)*zt(layer)) - m(j)**2*m(i)*1.0&
      /(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j&
      )**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer&
      )*m(j)**4)*at(layer)*zt(layer)*cos(m(i)*zt(layer))*sin(m(j)*zt(&
      layer)) - m(j)**3*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(&
      i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)&
      **2*m(i)**2 - zt(layer)*m(j)**4)*ab(layer)*zt(layer)*cos(m(j)*zt(&
      layer))*sin(m(i)*zt(layer)) + m(j)**3*1.0/(zb(layer)*m(i)**4 - 2*&
      zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4&
      + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)*at(layer)*zt(&
      layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer)) - zb(layer)*m(i)**&
      3*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(layer&
      )*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(&
      layer)*m(j)**4)*at(layer)*cos(m(i)*zt(layer))*sin(m(j)*zt(layer&
      )) + zb(layer)*m(j)*m(i)**2*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*&
      m(j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(&
      layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)*at(layer)*cos(m(j)*zt&
      (layer))*sin(m(i)*zt(layer)) + zb(layer)*m(j)**2*m(i)*1.0/(zb(&
      layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4&
      - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j&
      )**4)*at(layer)*cos(m(i)*zt(layer))*sin(m(j)*zt(layer)) - zb(&
      layer)*m(j)**3*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)&
      **2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2&
      *m(i)**2 - zt(layer)*m(j)**4)*at(layer)*cos(m(j)*zt(layer))*sin(m&
      (i)*zt(layer)) + zt(layer)*m(i)**3*1.0/(zb(layer)*m(i)**4 - 2*zb(&
      layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 +&
      2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)*ab(layer)*cos(m(&
      i)*zt(layer))*sin(m(j)*zt(layer)) - zt(layer)*m(j)*m(i)**2*1.0/(&
      zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)&
      **4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)&
      *m(j)**4)*ab(layer)*cos(m(j)*zt(layer))*sin(m(i)*zt(layer)) - zt(&
      layer)*m(j)**2*m(i)*1.0/(zb(layer)*m(i)**4 - 2*zb(layer)*m(j)**2*&
      m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)**4 + 2*zt(layer)*m(j&
      )**2*m(i)**2 - zt(layer)*m(j)**4)*ab(layer)*cos(m(i)*zt(layer))*&
      sin(m(j)*zt(layer)) + zt(layer)*m(j)**3*1.0/(zb(layer)*m(i)**4 -&
      2*zb(layer)*m(j)**2*m(i)**2 + zb(layer)*m(j)**4 - zt(layer)*m(i)&
      **4 + 2*zt(layer)*m(j)**2*m(i)**2 - zt(layer)*m(j)**4)*ab(layer)*&
      cos(m(j)*zt(layer))*sin(m(i)*zt(layer))))
            END DO
          END DO
        END DO

        DO j = 0, neig -2
          DO i = j + 1, neig-1
            a(j,i) = a(i, j)
          END DO
        END DO
                    
      END SUBROUTINE

      SUBROUTINE dim1sin_ab_linear(m, at, ab, bt, bb, zt, zb, a, neig, &
                    nlayers) 
        USE types
        IMPLICIT NONE     
        
        INTEGER, intent(in) :: neig        
        INTEGER, intent(in) :: nlayers
        REAL(DP), intent(in), dimension(0:neig-1) ::m
        REAL(DP), intent(in), dimension(0:nlayers-1) :: at,ab,bt,bb, &
                                                        zt,zb
        REAL(DP), intent(out), dimension(0:neig-1) :: a
        INTEGER :: i, layer
        a=0.0D0
        DO layer = 0, nlayers-1
          DO i = 0, neig-1
      a(i) = a(i) + (2*1.0/(zb(layer)**2*m(i)**3 - 2*zt(layer)*zb(layer)&
      *m(i)**3 + zt(layer)**2*m(i)**3)*bb(layer)*ab(layer)*cos(m(i)*zb(&
      layer)) - 2*1.0/(zb(layer)**2*m(i)**3 - 2*zt(layer)*zb(layer)*m(i&
      )**3 + zt(layer)**2*m(i)**3)*bb(layer)*ab(layer)*cos(m(i)*zt(&
      layer)) - 2*1.0/(zb(layer)**2*m(i)**3 - 2*zt(layer)*zb(layer)*m(i&
      )**3 + zt(layer)**2*m(i)**3)*bb(layer)*at(layer)*cos(m(i)*zb(&
      layer)) + 2*1.0/(zb(layer)**2*m(i)**3 - 2*zt(layer)*zb(layer)*m(i&
      )**3 + zt(layer)**2*m(i)**3)*bb(layer)*at(layer)*cos(m(i)*zt(&
      layer)) - 2*1.0/(zb(layer)**2*m(i)**3 - 2*zt(layer)*zb(layer)*m(i&
      )**3 + zt(layer)**2*m(i)**3)*bt(layer)*ab(layer)*cos(m(i)*zb(&
      layer)) + 2*1.0/(zb(layer)**2*m(i)**3 - 2*zt(layer)*zb(layer)*m(i&
      )**3 + zt(layer)**2*m(i)**3)*bt(layer)*ab(layer)*cos(m(i)*zt(&
      layer)) + 2*1.0/(zb(layer)**2*m(i)**3 - 2*zt(layer)*zb(layer)*m(i&
      )**3 + zt(layer)**2*m(i)**3)*bt(layer)*at(layer)*cos(m(i)*zb(&
      layer)) - 2*1.0/(zb(layer)**2*m(i)**3 - 2*zt(layer)*zb(layer)*m(i&
      )**3 + zt(layer)**2*m(i)**3)*bt(layer)*at(layer)*cos(m(i)*zt(&
      layer)) + 2*m(i)*1.0/(zb(layer)**2*m(i)**3 - 2*zt(layer)*zb(layer&
      )*m(i)**3 + zt(layer)**2*m(i)**3)*bb(layer)*ab(layer)*zb(layer)*&
      sin(m(i)*zb(layer)) - 2*m(i)*1.0/(zb(layer)**2*m(i)**3 - 2*zt(&
      layer)*zb(layer)*m(i)**3 + zt(layer)**2*m(i)**3)*bb(layer)*ab(&
      layer)*zt(layer)*sin(m(i)*zt(layer)) - 2*m(i)*1.0/(zb(layer)**2*m&
      (i)**3 - 2*zt(layer)*zb(layer)*m(i)**3 + zt(layer)**2*m(i)**3)*bb&
      (layer)*at(layer)*zb(layer)*sin(m(i)*zb(layer)) + 2*m(i)*1.0/(zb(&
      layer)**2*m(i)**3 - 2*zt(layer)*zb(layer)*m(i)**3 + zt(layer)**2*&
      m(i)**3)*bb(layer)*at(layer)*zt(layer)*sin(m(i)*zt(layer)) - 2*m(&
      i)*1.0/(zb(layer)**2*m(i)**3 - 2*zt(layer)*zb(layer)*m(i)**3 + zt&
      (layer)**2*m(i)**3)*bt(layer)*ab(layer)*zb(layer)*sin(m(i)*zb(&
      layer)) + 2*m(i)*1.0/(zb(layer)**2*m(i)**3 - 2*zt(layer)*zb(layer&
      )*m(i)**3 + zt(layer)**2*m(i)**3)*bt(layer)*ab(layer)*zt(layer)*&
      sin(m(i)*zt(layer)) + 2*m(i)*1.0/(zb(layer)**2*m(i)**3 - 2*zt(&
      layer)*zb(layer)*m(i)**3 + zt(layer)**2*m(i)**3)*bt(layer)*at(&
      layer)*zb(layer)*sin(m(i)*zb(layer)) - 2*m(i)*1.0/(zb(layer)**2*m&
      (i)**3 - 2*zt(layer)*zb(layer)*m(i)**3 + zt(layer)**2*m(i)**3)*bt&
      (layer)*at(layer)*zt(layer)*sin(m(i)*zt(layer)) - m(i)**2*1.0/(zb&
      (layer)**2*m(i)**3 - 2*zt(layer)*zb(layer)*m(i)**3 + zt(layer)**2&
      *m(i)**3)*bb(layer)*ab(layer)*zb(layer)**2*cos(m(i)*zb(layer)) +&
      m(i)**2*1.0/(zb(layer)**2*m(i)**3 - 2*zt(layer)*zb(layer)*m(i)**3&
      + zt(layer)**2*m(i)**3)*bb(layer)*ab(layer)*zt(layer)**2*cos(m(i)&
      *zt(layer)) + m(i)**2*1.0/(zb(layer)**2*m(i)**3 - 2*zt(layer)*zb(&
      layer)*m(i)**3 + zt(layer)**2*m(i)**3)*bb(layer)*at(layer)*zb(&
      layer)**2*cos(m(i)*zb(layer)) - m(i)**2*1.0/(zb(layer)**2*m(i)**3&
      - 2*zt(layer)*zb(layer)*m(i)**3 + zt(layer)**2*m(i)**3)*bb(layer)&
      *at(layer)*zt(layer)**2*cos(m(i)*zt(layer)) + m(i)**2*1.0/(zb(&
      layer)**2*m(i)**3 - 2*zt(layer)*zb(layer)*m(i)**3 + zt(layer)**2*&
      m(i)**3)*bt(layer)*ab(layer)*zb(layer)**2*cos(m(i)*zb(layer)) - m&
      (i)**2*1.0/(zb(layer)**2*m(i)**3 - 2*zt(layer)*zb(layer)*m(i)**3&
      + zt(layer)**2*m(i)**3)*bt(layer)*ab(layer)*zt(layer)**2*cos(m(i)&
      *zt(layer)) - m(i)**2*1.0/(zb(layer)**2*m(i)**3 - 2*zt(layer)*zb(&
      layer)*m(i)**3 + zt(layer)**2*m(i)**3)*bt(layer)*at(layer)*zb(&
      layer)**2*cos(m(i)*zb(layer)) + m(i)**2*1.0/(zb(layer)**2*m(i)**3&
      - 2*zt(layer)*zb(layer)*m(i)**3 + zt(layer)**2*m(i)**3)*bt(layer)&
      *at(layer)*zt(layer)**2*cos(m(i)*zt(layer)) + zb(layer)*m(i)*1.0/&
      (zb(layer)**2*m(i)**3 - 2*zt(layer)*zb(layer)*m(i)**3 + zt(layer)&
      **2*m(i)**3)*bb(layer)*at(layer)*sin(m(i)*zb(layer)) - zb(layer)*&
      m(i)*1.0/(zb(layer)**2*m(i)**3 - 2*zt(layer)*zb(layer)*m(i)**3 +&
      zt(layer)**2*m(i)**3)*bb(layer)*at(layer)*sin(m(i)*zt(layer)) +&
      zb(layer)*m(i)*1.0/(zb(layer)**2*m(i)**3 - 2*zt(layer)*zb(layer)*&
      m(i)**3 + zt(layer)**2*m(i)**3)*bt(layer)*ab(layer)*sin(m(i)*zb(&
      layer)) - zb(layer)*m(i)*1.0/(zb(layer)**2*m(i)**3 - 2*zt(layer)*&
      zb(layer)*m(i)**3 + zt(layer)**2*m(i)**3)*bt(layer)*ab(layer)*sin&
      (m(i)*zt(layer)) - 2*zb(layer)*m(i)*1.0/(zb(layer)**2*m(i)**3 - 2&
      *zt(layer)*zb(layer)*m(i)**3 + zt(layer)**2*m(i)**3)*bt(layer)*at&
      (layer)*sin(m(i)*zb(layer)) + 2*zb(layer)*m(i)*1.0/(zb(layer)**2*&
      m(i)**3 - 2*zt(layer)*zb(layer)*m(i)**3 + zt(layer)**2*m(i)**3)*&
      bt(layer)*at(layer)*sin(m(i)*zt(layer)) - zb(layer)*m(i)**2*1.0/(&
      zb(layer)**2*m(i)**3 - 2*zt(layer)*zb(layer)*m(i)**3 + zt(layer)&
      **2*m(i)**3)*bb(layer)*at(layer)*zb(layer)*cos(m(i)*zb(layer)) +&
      zb(layer)*m(i)**2*1.0/(zb(layer)**2*m(i)**3 - 2*zt(layer)*zb(&
      layer)*m(i)**3 + zt(layer)**2*m(i)**3)*bb(layer)*at(layer)*zt(&
      layer)*cos(m(i)*zt(layer)) - zb(layer)*m(i)**2*1.0/(zb(layer)**2*&
      m(i)**3 - 2*zt(layer)*zb(layer)*m(i)**3 + zt(layer)**2*m(i)**3)*&
      bt(layer)*ab(layer)*zb(layer)*cos(m(i)*zb(layer)) + zb(layer)*m(i&
      )**2*1.0/(zb(layer)**2*m(i)**3 - 2*zt(layer)*zb(layer)*m(i)**3 +&
      zt(layer)**2*m(i)**3)*bt(layer)*ab(layer)*zt(layer)*cos(m(i)*zt(&
      layer)) + 2*zb(layer)*m(i)**2*1.0/(zb(layer)**2*m(i)**3 - 2*zt(&
      layer)*zb(layer)*m(i)**3 + zt(layer)**2*m(i)**3)*bt(layer)*at(&
      layer)*zb(layer)*cos(m(i)*zb(layer)) - 2*zb(layer)*m(i)**2*1.0/(&
      zb(layer)**2*m(i)**3 - 2*zt(layer)*zb(layer)*m(i)**3 + zt(layer)&
      **2*m(i)**3)*bt(layer)*at(layer)*zt(layer)*cos(m(i)*zt(layer)) -&
      zb(layer)**2*m(i)**2*1.0/(zb(layer)**2*m(i)**3 - 2*zt(layer)*zb(&
      layer)*m(i)**3 + zt(layer)**2*m(i)**3)*bt(layer)*at(layer)*cos(m(&
      i)*zb(layer)) + zb(layer)**2*m(i)**2*1.0/(zb(layer)**2*m(i)**3 -&
      2*zt(layer)*zb(layer)*m(i)**3 + zt(layer)**2*m(i)**3)*bt(layer)*&
      at(layer)*cos(m(i)*zt(layer)) - 2*zt(layer)*m(i)*1.0/(zb(layer)**&
      2*m(i)**3 - 2*zt(layer)*zb(layer)*m(i)**3 + zt(layer)**2*m(i)**3)&
      *bb(layer)*ab(layer)*sin(m(i)*zb(layer)) + 2*zt(layer)*m(i)*1.0/(&
      zb(layer)**2*m(i)**3 - 2*zt(layer)*zb(layer)*m(i)**3 + zt(layer)&
      **2*m(i)**3)*bb(layer)*ab(layer)*sin(m(i)*zt(layer)) + zt(layer)*&
      m(i)*1.0/(zb(layer)**2*m(i)**3 - 2*zt(layer)*zb(layer)*m(i)**3 +&
      zt(layer)**2*m(i)**3)*bb(layer)*at(layer)*sin(m(i)*zb(layer)) -&
      zt(layer)*m(i)*1.0/(zb(layer)**2*m(i)**3 - 2*zt(layer)*zb(layer)*&
      m(i)**3 + zt(layer)**2*m(i)**3)*bb(layer)*at(layer)*sin(m(i)*zt(&
      layer)) + zt(layer)*m(i)*1.0/(zb(layer)**2*m(i)**3 - 2*zt(layer)*&
      zb(layer)*m(i)**3 + zt(layer)**2*m(i)**3)*bt(layer)*ab(layer)*sin&
      (m(i)*zb(layer)) - zt(layer)*m(i)*1.0/(zb(layer)**2*m(i)**3 - 2*&
      zt(layer)*zb(layer)*m(i)**3 + zt(layer)**2*m(i)**3)*bt(layer)*ab(&
      layer)*sin(m(i)*zt(layer)) + 2*zt(layer)*m(i)**2*1.0/(zb(layer)**&
      2*m(i)**3 - 2*zt(layer)*zb(layer)*m(i)**3 + zt(layer)**2*m(i)**3)&
      *bb(layer)*ab(layer)*zb(layer)*cos(m(i)*zb(layer)) - 2*zt(layer)*&
      m(i)**2*1.0/(zb(layer)**2*m(i)**3 - 2*zt(layer)*zb(layer)*m(i)**3&
      + zt(layer)**2*m(i)**3)*bb(layer)*ab(layer)*zt(layer)*cos(m(i)*zt&
      (layer)) - zt(layer)*m(i)**2*1.0/(zb(layer)**2*m(i)**3 - 2*zt(&
      layer)*zb(layer)*m(i)**3 + zt(layer)**2*m(i)**3)*bb(layer)*at(&
      layer)*zb(layer)*cos(m(i)*zb(layer)) + zt(layer)*m(i)**2*1.0/(zb(&
      layer)**2*m(i)**3 - 2*zt(layer)*zb(layer)*m(i)**3 + zt(layer)**2*&
      m(i)**3)*bb(layer)*at(layer)*zt(layer)*cos(m(i)*zt(layer)) - zt(&
      layer)*m(i)**2*1.0/(zb(layer)**2*m(i)**3 - 2*zt(layer)*zb(layer)*&
      m(i)**3 + zt(layer)**2*m(i)**3)*bt(layer)*ab(layer)*zb(layer)*cos&
      (m(i)*zb(layer)) + zt(layer)*m(i)**2*1.0/(zb(layer)**2*m(i)**3 -&
      2*zt(layer)*zb(layer)*m(i)**3 + zt(layer)**2*m(i)**3)*bt(layer)*&
      ab(layer)*zt(layer)*cos(m(i)*zt(layer)) + zt(layer)*zb(layer)*m(i&
      )**2*1.0/(zb(layer)**2*m(i)**3 - 2*zt(layer)*zb(layer)*m(i)**3 +&
      zt(layer)**2*m(i)**3)*bb(layer)*at(layer)*cos(m(i)*zb(layer)) -&
      zt(layer)*zb(layer)*m(i)**2*1.0/(zb(layer)**2*m(i)**3 - 2*zt(&
      layer)*zb(layer)*m(i)**3 + zt(layer)**2*m(i)**3)*bb(layer)*at(&
      layer)*cos(m(i)*zt(layer)) + zt(layer)*zb(layer)*m(i)**2*1.0/(zb(&
      layer)**2*m(i)**3 - 2*zt(layer)*zb(layer)*m(i)**3 + zt(layer)**2*&
      m(i)**3)*bt(layer)*ab(layer)*cos(m(i)*zb(layer)) - zt(layer)*zb(&
      layer)*m(i)**2*1.0/(zb(layer)**2*m(i)**3 - 2*zt(layer)*zb(layer)*&
      m(i)**3 + zt(layer)**2*m(i)**3)*bt(layer)*ab(layer)*cos(m(i)*zt(&
      layer)) - zt(layer)**2*m(i)**2*1.0/(zb(layer)**2*m(i)**3 - 2*zt(&
      layer)*zb(layer)*m(i)**3 + zt(layer)**2*m(i)**3)*bb(layer)*ab(&
      layer)*cos(m(i)*zb(layer)) + zt(layer)**2*m(i)**2*1.0/(zb(layer)&
      **2*m(i)**3 - 2*zt(layer)*zb(layer)*m(i)**3 + zt(layer)**2*m(i)**&
      3)*bb(layer)*ab(layer)*cos(m(i)*zt(layer)))
          END DO         
        END DO    
                    
      END SUBROUTINE
      
      SUBROUTINE dim1sin_abc_linear(m, at, ab, bt, bb, ct, cb, zt, zb, &
                 a, neig, nlayers) 
        USE types
        IMPLICIT NONE     
        
        INTEGER, intent(in) :: neig        
        INTEGER, intent(in) :: nlayers
        REAL(DP), intent(in), dimension(0:neig-1) ::m
        REAL(DP), intent(in), dimension(0:nlayers-1) :: at,ab,bt,bb,ct,&
                                                        cb, zt,zb
        REAL(DP), intent(out), dimension(0:neig-1) :: a
        INTEGER :: i, layer
        a=0.0D0
        DO layer = 0, nlayers-1
          DO i = 0, neig-1
      a(i) = a(i) + (-6*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer&
      )**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(&
      i)**4)*cb(layer)*bb(layer)*ab(layer)*sin(m(i)*zb(layer)) + 6*1.0/&
      (zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(&
      layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bb(&
      layer)*ab(layer)*sin(m(i)*zt(layer)) + 6*1.0/(zb(layer)**3*m(i)**&
      4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m&
      (i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bb(layer)*at(layer)*sin(&
      m(i)*zb(layer)) - 6*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(&
      layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)&
      **3*m(i)**4)*cb(layer)*bb(layer)*at(layer)*sin(m(i)*zt(layer)) +&
      6*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 +&
      3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer&
      )*bt(layer)*ab(layer)*sin(m(i)*zb(layer)) - 6*1.0/(zb(layer)**3*m&
      (i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(&
      layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bt(layer)*ab(&
      layer)*sin(m(i)*zt(layer)) - 6*1.0/(zb(layer)**3*m(i)**4 - 3*zt(&
      layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 -&
      zt(layer)**3*m(i)**4)*cb(layer)*bt(layer)*at(layer)*sin(m(i)*zb(&
      layer)) + 6*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*&
      m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4&
      )*cb(layer)*bt(layer)*at(layer)*sin(m(i)*zt(layer)) + 6*1.0/(zb(&
      layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer&
      )**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(layer&
      )*ab(layer)*sin(m(i)*zb(layer)) - 6*1.0/(zb(layer)**3*m(i)**4 - 3&
      *zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**&
      4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(layer)*ab(layer)*sin(m(i)*&
      zt(layer)) - 6*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)&
      **2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i&
      )**4)*ct(layer)*bb(layer)*at(layer)*sin(m(i)*zb(layer)) + 6*1.0/(&
      zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(&
      layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(&
      layer)*at(layer)*sin(m(i)*zt(layer)) - 6*1.0/(zb(layer)**3*m(i)**&
      4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m&
      (i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bt(layer)*ab(layer)*sin(&
      m(i)*zb(layer)) + 6*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(&
      layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)&
      **3*m(i)**4)*ct(layer)*bt(layer)*ab(layer)*sin(m(i)*zt(layer)) +&
      6*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 +&
      3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer&
      )*bt(layer)*at(layer)*sin(m(i)*zb(layer)) - 6*1.0/(zb(layer)**3*m&
      (i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(&
      layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bt(layer)*at(&
      layer)*sin(m(i)*zt(layer)) + 6*m(i)*1.0/(zb(layer)**3*m(i)**4 - 3&
      *zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**&
      4 - zt(layer)**3*m(i)**4)*cb(layer)*bb(layer)*ab(layer)*zb(layer)&
      *cos(m(i)*zb(layer)) - 6*m(i)*1.0/(zb(layer)**3*m(i)**4 - 3*zt(&
      layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 -&
      zt(layer)**3*m(i)**4)*cb(layer)*bb(layer)*ab(layer)*zt(layer)*cos&
      (m(i)*zt(layer)) - 6*m(i)*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)&
      *zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(&
      layer)**3*m(i)**4)*cb(layer)*bb(layer)*at(layer)*zb(layer)*cos(m(&
      i)*zb(layer)) + 6*m(i)*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb&
      (layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)&
      **3*m(i)**4)*cb(layer)*bb(layer)*at(layer)*zt(layer)*cos(m(i)*zt(&
      layer)) - 6*m(i)*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer&
      )**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(&
      i)**4)*cb(layer)*bt(layer)*ab(layer)*zb(layer)*cos(m(i)*zb(layer&
      )) + 6*m(i)*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*&
      m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4&
      )*cb(layer)*bt(layer)*ab(layer)*zt(layer)*cos(m(i)*zt(layer)) + 6&
      *m(i)*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**&
      4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(&
      layer)*bt(layer)*at(layer)*zb(layer)*cos(m(i)*zb(layer)) - 6*m(i)&
      *1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3&
      *zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)&
      *bt(layer)*at(layer)*zt(layer)*cos(m(i)*zt(layer)) - 6*m(i)*1.0/(&
      zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(&
      layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(&
      layer)*ab(layer)*zb(layer)*cos(m(i)*zb(layer)) + 6*m(i)*1.0/(zb(&
      layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer&
      )**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(layer&
      )*ab(layer)*zt(layer)*cos(m(i)*zt(layer)) + 6*m(i)*1.0/(zb(layer)&
      **3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(layer)*at(&
      layer)*zb(layer)*cos(m(i)*zb(layer)) - 6*m(i)*1.0/(zb(layer)**3*m&
      (i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(&
      layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(layer)*at(&
      layer)*zt(layer)*cos(m(i)*zt(layer)) + 6*m(i)*1.0/(zb(layer)**3*m&
      (i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(&
      layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bt(layer)*ab(&
      layer)*zb(layer)*cos(m(i)*zb(layer)) - 6*m(i)*1.0/(zb(layer)**3*m&
      (i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(&
      layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bt(layer)*ab(&
      layer)*zt(layer)*cos(m(i)*zt(layer)) - 6*m(i)*1.0/(zb(layer)**3*m&
      (i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(&
      layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bt(layer)*at(&
      layer)*zb(layer)*cos(m(i)*zb(layer)) + 6*m(i)*1.0/(zb(layer)**3*m&
      (i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(&
      layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bt(layer)*at(&
      layer)*zt(layer)*cos(m(i)*zt(layer)) + 3*m(i)**2*1.0/(zb(layer)**&
      3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(&
      layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bb(layer)*ab(&
      layer)*zb(layer)**2*sin(m(i)*zb(layer)) - 3*m(i)**2*1.0/(zb(layer&
      )**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bb(layer)*ab(&
      layer)*zt(layer)**2*sin(m(i)*zt(layer)) - 3*m(i)**2*1.0/(zb(layer&
      )**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bb(layer)*at(&
      layer)*zb(layer)**2*sin(m(i)*zb(layer)) + 3*m(i)**2*1.0/(zb(layer&
      )**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bb(layer)*at(&
      layer)*zt(layer)**2*sin(m(i)*zt(layer)) - 3*m(i)**2*1.0/(zb(layer&
      )**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bt(layer)*ab(&
      layer)*zb(layer)**2*sin(m(i)*zb(layer)) + 3*m(i)**2*1.0/(zb(layer&
      )**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bt(layer)*ab(&
      layer)*zt(layer)**2*sin(m(i)*zt(layer)) + 3*m(i)**2*1.0/(zb(layer&
      )**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bt(layer)*at(&
      layer)*zb(layer)**2*sin(m(i)*zb(layer)) - 3*m(i)**2*1.0/(zb(layer&
      )**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bt(layer)*at(&
      layer)*zt(layer)**2*sin(m(i)*zt(layer)) - 3*m(i)**2*1.0/(zb(layer&
      )**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(layer)*ab(&
      layer)*zb(layer)**2*sin(m(i)*zb(layer)) + 3*m(i)**2*1.0/(zb(layer&
      )**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(layer)*ab(&
      layer)*zt(layer)**2*sin(m(i)*zt(layer)) + 3*m(i)**2*1.0/(zb(layer&
      )**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(layer)*at(&
      layer)*zb(layer)**2*sin(m(i)*zb(layer)) - 3*m(i)**2*1.0/(zb(layer&
      )**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(layer)*at(&
      layer)*zt(layer)**2*sin(m(i)*zt(layer)) + 3*m(i)**2*1.0/(zb(layer&
      )**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bt(layer)*ab(&
      layer)*zb(layer)**2*sin(m(i)*zb(layer)) - 3*m(i)**2*1.0/(zb(layer&
      )**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bt(layer)*ab(&
      layer)*zt(layer)**2*sin(m(i)*zt(layer)) - 3*m(i)**2*1.0/(zb(layer&
      )**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bt(layer)*at(&
      layer)*zb(layer)**2*sin(m(i)*zb(layer)) + 3*m(i)**2*1.0/(zb(layer&
      )**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bt(layer)*at(&
      layer)*zt(layer)**2*sin(m(i)*zt(layer)) - m(i)**3*1.0/(zb(layer)&
      **3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bb(layer)*ab(&
      layer)*zb(layer)**3*cos(m(i)*zb(layer)) + m(i)**3*1.0/(zb(layer)&
      **3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bb(layer)*ab(&
      layer)*zt(layer)**3*cos(m(i)*zt(layer)) + m(i)**3*1.0/(zb(layer)&
      **3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bb(layer)*at(&
      layer)*zb(layer)**3*cos(m(i)*zb(layer)) - m(i)**3*1.0/(zb(layer)&
      **3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bb(layer)*at(&
      layer)*zt(layer)**3*cos(m(i)*zt(layer)) + m(i)**3*1.0/(zb(layer)&
      **3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bt(layer)*ab(&
      layer)*zb(layer)**3*cos(m(i)*zb(layer)) - m(i)**3*1.0/(zb(layer)&
      **3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bt(layer)*ab(&
      layer)*zt(layer)**3*cos(m(i)*zt(layer)) - m(i)**3*1.0/(zb(layer)&
      **3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bt(layer)*at(&
      layer)*zb(layer)**3*cos(m(i)*zb(layer)) + m(i)**3*1.0/(zb(layer)&
      **3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bt(layer)*at(&
      layer)*zt(layer)**3*cos(m(i)*zt(layer)) + m(i)**3*1.0/(zb(layer)&
      **3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(layer)*ab(&
      layer)*zb(layer)**3*cos(m(i)*zb(layer)) - m(i)**3*1.0/(zb(layer)&
      **3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(layer)*ab(&
      layer)*zt(layer)**3*cos(m(i)*zt(layer)) - m(i)**3*1.0/(zb(layer)&
      **3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(layer)*at(&
      layer)*zb(layer)**3*cos(m(i)*zb(layer)) + m(i)**3*1.0/(zb(layer)&
      **3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(layer)*at(&
      layer)*zt(layer)**3*cos(m(i)*zt(layer)) - m(i)**3*1.0/(zb(layer)&
      **3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bt(layer)*ab(&
      layer)*zb(layer)**3*cos(m(i)*zb(layer)) + m(i)**3*1.0/(zb(layer)&
      **3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bt(layer)*ab(&
      layer)*zt(layer)**3*cos(m(i)*zt(layer)) + m(i)**3*1.0/(zb(layer)&
      **3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bt(layer)*at(&
      layer)*zb(layer)**3*cos(m(i)*zb(layer)) - m(i)**3*1.0/(zb(layer)&
      **3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bt(layer)*at(&
      layer)*zt(layer)**3*cos(m(i)*zt(layer)) + 2*zb(layer)*m(i)*1.0/(&
      zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(&
      layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bb(&
      layer)*at(layer)*cos(m(i)*zb(layer)) - 2*zb(layer)*m(i)*1.0/(zb(&
      layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer&
      )**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bb(layer&
      )*at(layer)*cos(m(i)*zt(layer)) + 2*zb(layer)*m(i)*1.0/(zb(layer)&
      **3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bt(layer)*ab(&
      layer)*cos(m(i)*zb(layer)) - 2*zb(layer)*m(i)*1.0/(zb(layer)**3*m&
      (i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(&
      layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bt(layer)*ab(&
      layer)*cos(m(i)*zt(layer)) - 4*zb(layer)*m(i)*1.0/(zb(layer)**3*m&
      (i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(&
      layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bt(layer)*at(&
      layer)*cos(m(i)*zb(layer)) + 4*zb(layer)*m(i)*1.0/(zb(layer)**3*m&
      (i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(&
      layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bt(layer)*at(&
      layer)*cos(m(i)*zt(layer)) + 2*zb(layer)*m(i)*1.0/(zb(layer)**3*m&
      (i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(&
      layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(layer)*ab(&
      layer)*cos(m(i)*zb(layer)) - 2*zb(layer)*m(i)*1.0/(zb(layer)**3*m&
      (i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(&
      layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(layer)*ab(&
      layer)*cos(m(i)*zt(layer)) - 4*zb(layer)*m(i)*1.0/(zb(layer)**3*m&
      (i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(&
      layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(layer)*at(&
      layer)*cos(m(i)*zb(layer)) + 4*zb(layer)*m(i)*1.0/(zb(layer)**3*m&
      (i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(&
      layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(layer)*at(&
      layer)*cos(m(i)*zt(layer)) - 4*zb(layer)*m(i)*1.0/(zb(layer)**3*m&
      (i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(&
      layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bt(layer)*ab(&
      layer)*cos(m(i)*zb(layer)) + 4*zb(layer)*m(i)*1.0/(zb(layer)**3*m&
      (i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(&
      layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bt(layer)*ab(&
      layer)*cos(m(i)*zt(layer)) + 6*zb(layer)*m(i)*1.0/(zb(layer)**3*m&
      (i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(&
      layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bt(layer)*at(&
      layer)*cos(m(i)*zb(layer)) - 6*zb(layer)*m(i)*1.0/(zb(layer)**3*m&
      (i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(&
      layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bt(layer)*at(&
      layer)*cos(m(i)*zt(layer)) + 2*zb(layer)*m(i)**2*1.0/(zb(layer)**&
      3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(&
      layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bb(layer)*at(&
      layer)*zb(layer)*sin(m(i)*zb(layer)) - 2*zb(layer)*m(i)**2*1.0/(&
      zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(&
      layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bb(&
      layer)*at(layer)*zt(layer)*sin(m(i)*zt(layer)) + 2*zb(layer)*m(i)&
      **2*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4&
      + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(&
      layer)*bt(layer)*ab(layer)*zb(layer)*sin(m(i)*zb(layer)) - 2*zb(&
      layer)*m(i)**2*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)&
      **2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i&
      )**4)*cb(layer)*bt(layer)*ab(layer)*zt(layer)*sin(m(i)*zt(layer&
      )) - 4*zb(layer)*m(i)**2*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*&
      zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(&
      layer)**3*m(i)**4)*cb(layer)*bt(layer)*at(layer)*zb(layer)*sin(m(&
      i)*zb(layer)) + 4*zb(layer)*m(i)**2*1.0/(zb(layer)**3*m(i)**4 - 3&
      *zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**&
      4 - zt(layer)**3*m(i)**4)*cb(layer)*bt(layer)*at(layer)*zt(layer)&
      *sin(m(i)*zt(layer)) + 2*zb(layer)*m(i)**2*1.0/(zb(layer)**3*m(i)&
      **4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)&
      *m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(layer)*ab(layer)*zb&
      (layer)*sin(m(i)*zb(layer)) - 2*zb(layer)*m(i)**2*1.0/(zb(layer)&
      **3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(layer)*ab(&
      layer)*zt(layer)*sin(m(i)*zt(layer)) - 4*zb(layer)*m(i)**2*1.0/(&
      zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(&
      layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(&
      layer)*at(layer)*zb(layer)*sin(m(i)*zb(layer)) + 4*zb(layer)*m(i)&
      **2*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4&
      + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(&
      layer)*bb(layer)*at(layer)*zt(layer)*sin(m(i)*zt(layer)) - 4*zb(&
      layer)*m(i)**2*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)&
      **2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i&
      )**4)*ct(layer)*bt(layer)*ab(layer)*zb(layer)*sin(m(i)*zb(layer&
      )) + 4*zb(layer)*m(i)**2*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*&
      zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(&
      layer)**3*m(i)**4)*ct(layer)*bt(layer)*ab(layer)*zt(layer)*sin(m(&
      i)*zt(layer)) + 6*zb(layer)*m(i)**2*1.0/(zb(layer)**3*m(i)**4 - 3&
      *zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**&
      4 - zt(layer)**3*m(i)**4)*ct(layer)*bt(layer)*at(layer)*zb(layer)&
      *sin(m(i)*zb(layer)) - 6*zb(layer)*m(i)**2*1.0/(zb(layer)**3*m(i)&
      **4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)&
      *m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bt(layer)*at(layer)*zt&
      (layer)*sin(m(i)*zt(layer)) - zb(layer)*m(i)**3*1.0/(zb(layer)**3&
      *m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(&
      layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bb(layer)*at(&
      layer)*zb(layer)**2*cos(m(i)*zb(layer)) + zb(layer)*m(i)**3*1.0/(&
      zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(&
      layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bb(&
      layer)*at(layer)*zt(layer)**2*cos(m(i)*zt(layer)) - zb(layer)*m(i&
      )**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4&
      + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(&
      layer)*bt(layer)*ab(layer)*zb(layer)**2*cos(m(i)*zb(layer)) + zb(&
      layer)*m(i)**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)&
      **2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i&
      )**4)*cb(layer)*bt(layer)*ab(layer)*zt(layer)**2*cos(m(i)*zt(&
      layer)) + 2*zb(layer)*m(i)**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(&
      layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 -&
      zt(layer)**3*m(i)**4)*cb(layer)*bt(layer)*at(layer)*zb(layer)**2*&
      cos(m(i)*zb(layer)) - 2*zb(layer)*m(i)**3*1.0/(zb(layer)**3*m(i)&
      **4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)&
      *m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bt(layer)*at(layer)*zt&
      (layer)**2*cos(m(i)*zt(layer)) - zb(layer)*m(i)**3*1.0/(zb(layer)&
      **3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(layer)*ab(&
      layer)*zb(layer)**2*cos(m(i)*zb(layer)) + zb(layer)*m(i)**3*1.0/(&
      zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(&
      layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(&
      layer)*ab(layer)*zt(layer)**2*cos(m(i)*zt(layer)) + 2*zb(layer)*m&
      (i)**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)&
      **4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct&
      (layer)*bb(layer)*at(layer)*zb(layer)**2*cos(m(i)*zb(layer)) - 2*&
      zb(layer)*m(i)**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(&
      layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)&
      **3*m(i)**4)*ct(layer)*bb(layer)*at(layer)*zt(layer)**2*cos(m(i)*&
      zt(layer)) + 2*zb(layer)*m(i)**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt&
      (layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 -&
      zt(layer)**3*m(i)**4)*ct(layer)*bt(layer)*ab(layer)*zb(layer)**2*&
      cos(m(i)*zb(layer)) - 2*zb(layer)*m(i)**3*1.0/(zb(layer)**3*m(i)&
      **4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)&
      *m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bt(layer)*ab(layer)*zt&
      (layer)**2*cos(m(i)*zt(layer)) - 3*zb(layer)*m(i)**3*1.0/(zb(&
      layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer&
      )**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bt(layer&
      )*at(layer)*zb(layer)**2*cos(m(i)*zb(layer)) + 3*zb(layer)*m(i)**&
      3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 +&
      3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer&
      )*bt(layer)*at(layer)*zt(layer)**2*cos(m(i)*zt(layer)) + zb(layer&
      )**2*m(i)**2*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2&
      *m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**&
      4)*cb(layer)*bt(layer)*at(layer)*sin(m(i)*zb(layer)) - zb(layer)&
      **2*m(i)**2*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*&
      m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4&
      )*cb(layer)*bt(layer)*at(layer)*sin(m(i)*zt(layer)) + zb(layer)**&
      2*m(i)**2*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(&
      i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*&
      ct(layer)*bb(layer)*at(layer)*sin(m(i)*zb(layer)) - zb(layer)**2*&
      m(i)**2*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)&
      **4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct&
      (layer)*bb(layer)*at(layer)*sin(m(i)*zt(layer)) + zb(layer)**2*m(&
      i)**2*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**&
      4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(&
      layer)*bt(layer)*ab(layer)*sin(m(i)*zb(layer)) - zb(layer)**2*m(i&
      )**2*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4&
      + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(&
      layer)*bt(layer)*ab(layer)*sin(m(i)*zt(layer)) - 3*zb(layer)**2*m&
      (i)**2*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)&
      **4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct&
      (layer)*bt(layer)*at(layer)*sin(m(i)*zb(layer)) + 3*zb(layer)**2*&
      m(i)**2*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)&
      **4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct&
      (layer)*bt(layer)*at(layer)*sin(m(i)*zt(layer)) - zb(layer)**2*m(&
      i)**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**&
      4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(&
      layer)*bt(layer)*at(layer)*zb(layer)*cos(m(i)*zb(layer)) + zb(&
      layer)**2*m(i)**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(&
      layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)&
      **3*m(i)**4)*cb(layer)*bt(layer)*at(layer)*zt(layer)*cos(m(i)*zt(&
      layer)) - zb(layer)**2*m(i)**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(&
      layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 -&
      zt(layer)**3*m(i)**4)*ct(layer)*bb(layer)*at(layer)*zb(layer)*cos&
      (m(i)*zb(layer)) + zb(layer)**2*m(i)**3*1.0/(zb(layer)**3*m(i)**4&
      - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i&
      )**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(layer)*at(layer)*zt(&
      layer)*cos(m(i)*zt(layer)) - zb(layer)**2*m(i)**3*1.0/(zb(layer)&
      **3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bt(layer)*ab(&
      layer)*zb(layer)*cos(m(i)*zb(layer)) + zb(layer)**2*m(i)**3*1.0/(&
      zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(&
      layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bt(&
      layer)*ab(layer)*zt(layer)*cos(m(i)*zt(layer)) + 3*zb(layer)**2*m&
      (i)**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)&
      **4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct&
      (layer)*bt(layer)*at(layer)*zb(layer)*cos(m(i)*zb(layer)) - 3*zb(&
      layer)**2*m(i)**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(&
      layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)&
      **3*m(i)**4)*ct(layer)*bt(layer)*at(layer)*zt(layer)*cos(m(i)*zt(&
      layer)) - zb(layer)**3*m(i)**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(&
      layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 -&
      zt(layer)**3*m(i)**4)*ct(layer)*bt(layer)*at(layer)*cos(m(i)*zb(&
      layer)) + zb(layer)**3*m(i)**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(&
      layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 -&
      zt(layer)**3*m(i)**4)*ct(layer)*bt(layer)*at(layer)*cos(m(i)*zt(&
      layer)) - 6*zt(layer)*m(i)*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer&
      )*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(&
      layer)**3*m(i)**4)*cb(layer)*bb(layer)*ab(layer)*cos(m(i)*zb(&
      layer)) + 6*zt(layer)*m(i)*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer&
      )*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(&
      layer)**3*m(i)**4)*cb(layer)*bb(layer)*ab(layer)*cos(m(i)*zt(&
      layer)) + 4*zt(layer)*m(i)*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer&
      )*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(&
      layer)**3*m(i)**4)*cb(layer)*bb(layer)*at(layer)*cos(m(i)*zb(&
      layer)) - 4*zt(layer)*m(i)*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer&
      )*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(&
      layer)**3*m(i)**4)*cb(layer)*bb(layer)*at(layer)*cos(m(i)*zt(&
      layer)) + 4*zt(layer)*m(i)*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer&
      )*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(&
      layer)**3*m(i)**4)*cb(layer)*bt(layer)*ab(layer)*cos(m(i)*zb(&
      layer)) - 4*zt(layer)*m(i)*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer&
      )*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(&
      layer)**3*m(i)**4)*cb(layer)*bt(layer)*ab(layer)*cos(m(i)*zt(&
      layer)) - 2*zt(layer)*m(i)*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer&
      )*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(&
      layer)**3*m(i)**4)*cb(layer)*bt(layer)*at(layer)*cos(m(i)*zb(&
      layer)) + 2*zt(layer)*m(i)*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer&
      )*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(&
      layer)**3*m(i)**4)*cb(layer)*bt(layer)*at(layer)*cos(m(i)*zt(&
      layer)) + 4*zt(layer)*m(i)*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer&
      )*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(&
      layer)**3*m(i)**4)*ct(layer)*bb(layer)*ab(layer)*cos(m(i)*zb(&
      layer)) - 4*zt(layer)*m(i)*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer&
      )*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(&
      layer)**3*m(i)**4)*ct(layer)*bb(layer)*ab(layer)*cos(m(i)*zt(&
      layer)) - 2*zt(layer)*m(i)*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer&
      )*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(&
      layer)**3*m(i)**4)*ct(layer)*bb(layer)*at(layer)*cos(m(i)*zb(&
      layer)) + 2*zt(layer)*m(i)*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer&
      )*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(&
      layer)**3*m(i)**4)*ct(layer)*bb(layer)*at(layer)*cos(m(i)*zt(&
      layer)) - 2*zt(layer)*m(i)*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer&
      )*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(&
      layer)**3*m(i)**4)*ct(layer)*bt(layer)*ab(layer)*cos(m(i)*zb(&
      layer)) + 2*zt(layer)*m(i)*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer&
      )*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(&
      layer)**3*m(i)**4)*ct(layer)*bt(layer)*ab(layer)*cos(m(i)*zt(&
      layer)) - 6*zt(layer)*m(i)**2*1.0/(zb(layer)**3*m(i)**4 - 3*zt(&
      layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 -&
      zt(layer)**3*m(i)**4)*cb(layer)*bb(layer)*ab(layer)*zb(layer)*sin&
      (m(i)*zb(layer)) + 6*zt(layer)*m(i)**2*1.0/(zb(layer)**3*m(i)**4&
      - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i&
      )**4 - zt(layer)**3*m(i)**4)*cb(layer)*bb(layer)*ab(layer)*zt(&
      layer)*sin(m(i)*zt(layer)) + 4*zt(layer)*m(i)**2*1.0/(zb(layer)**&
      3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(&
      layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bb(layer)*at(&
      layer)*zb(layer)*sin(m(i)*zb(layer)) - 4*zt(layer)*m(i)**2*1.0/(&
      zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(&
      layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bb(&
      layer)*at(layer)*zt(layer)*sin(m(i)*zt(layer)) + 4*zt(layer)*m(i)&
      **2*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4&
      + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(&
      layer)*bt(layer)*ab(layer)*zb(layer)*sin(m(i)*zb(layer)) - 4*zt(&
      layer)*m(i)**2*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)&
      **2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i&
      )**4)*cb(layer)*bt(layer)*ab(layer)*zt(layer)*sin(m(i)*zt(layer&
      )) - 2*zt(layer)*m(i)**2*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*&
      zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(&
      layer)**3*m(i)**4)*cb(layer)*bt(layer)*at(layer)*zb(layer)*sin(m(&
      i)*zb(layer)) + 2*zt(layer)*m(i)**2*1.0/(zb(layer)**3*m(i)**4 - 3&
      *zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**&
      4 - zt(layer)**3*m(i)**4)*cb(layer)*bt(layer)*at(layer)*zt(layer)&
      *sin(m(i)*zt(layer)) + 4*zt(layer)*m(i)**2*1.0/(zb(layer)**3*m(i)&
      **4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)&
      *m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(layer)*ab(layer)*zb&
      (layer)*sin(m(i)*zb(layer)) - 4*zt(layer)*m(i)**2*1.0/(zb(layer)&
      **3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(layer)*ab(&
      layer)*zt(layer)*sin(m(i)*zt(layer)) - 2*zt(layer)*m(i)**2*1.0/(&
      zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(&
      layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(&
      layer)*at(layer)*zb(layer)*sin(m(i)*zb(layer)) + 2*zt(layer)*m(i)&
      **2*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4&
      + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(&
      layer)*bb(layer)*at(layer)*zt(layer)*sin(m(i)*zt(layer)) - 2*zt(&
      layer)*m(i)**2*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)&
      **2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i&
      )**4)*ct(layer)*bt(layer)*ab(layer)*zb(layer)*sin(m(i)*zb(layer&
      )) + 2*zt(layer)*m(i)**2*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*&
      zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(&
      layer)**3*m(i)**4)*ct(layer)*bt(layer)*ab(layer)*zt(layer)*sin(m(&
      i)*zt(layer)) + 3*zt(layer)*m(i)**3*1.0/(zb(layer)**3*m(i)**4 - 3&
      *zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**&
      4 - zt(layer)**3*m(i)**4)*cb(layer)*bb(layer)*ab(layer)*zb(layer)&
      **2*cos(m(i)*zb(layer)) - 3*zt(layer)*m(i)**3*1.0/(zb(layer)**3*m&
      (i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(&
      layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bb(layer)*ab(&
      layer)*zt(layer)**2*cos(m(i)*zt(layer)) - 2*zt(layer)*m(i)**3*1.0&
      /(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(&
      layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bb(&
      layer)*at(layer)*zb(layer)**2*cos(m(i)*zb(layer)) + 2*zt(layer)*m&
      (i)**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)&
      **4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb&
      (layer)*bb(layer)*at(layer)*zt(layer)**2*cos(m(i)*zt(layer)) - 2*&
      zt(layer)*m(i)**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(&
      layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)&
      **3*m(i)**4)*cb(layer)*bt(layer)*ab(layer)*zb(layer)**2*cos(m(i)*&
      zb(layer)) + 2*zt(layer)*m(i)**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt&
      (layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 -&
      zt(layer)**3*m(i)**4)*cb(layer)*bt(layer)*ab(layer)*zt(layer)**2*&
      cos(m(i)*zt(layer)) + zt(layer)*m(i)**3*1.0/(zb(layer)**3*m(i)**4&
      - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i&
      )**4 - zt(layer)**3*m(i)**4)*cb(layer)*bt(layer)*at(layer)*zb(&
      layer)**2*cos(m(i)*zb(layer)) - zt(layer)*m(i)**3*1.0/(zb(layer)&
      **3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bt(layer)*at(&
      layer)*zt(layer)**2*cos(m(i)*zt(layer)) - 2*zt(layer)*m(i)**3*1.0&
      /(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(&
      layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(&
      layer)*ab(layer)*zb(layer)**2*cos(m(i)*zb(layer)) + 2*zt(layer)*m&
      (i)**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)&
      **4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct&
      (layer)*bb(layer)*ab(layer)*zt(layer)**2*cos(m(i)*zt(layer)) + zt&
      (layer)*m(i)**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)&
      **2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i&
      )**4)*ct(layer)*bb(layer)*at(layer)*zb(layer)**2*cos(m(i)*zb(&
      layer)) - zt(layer)*m(i)**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(&
      layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 -&
      zt(layer)**3*m(i)**4)*ct(layer)*bb(layer)*at(layer)*zt(layer)**2*&
      cos(m(i)*zt(layer)) + zt(layer)*m(i)**3*1.0/(zb(layer)**3*m(i)**4&
      - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i&
      )**4 - zt(layer)**3*m(i)**4)*ct(layer)*bt(layer)*ab(layer)*zb(&
      layer)**2*cos(m(i)*zb(layer)) - zt(layer)*m(i)**3*1.0/(zb(layer)&
      **3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bt(layer)*ab(&
      layer)*zt(layer)**2*cos(m(i)*zt(layer)) - 2*zt(layer)*zb(layer)*m&
      (i)**2*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)&
      **4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb&
      (layer)*bb(layer)*at(layer)*sin(m(i)*zb(layer)) + 2*zt(layer)*zb(&
      layer)*m(i)**2*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)&
      **2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i&
      )**4)*cb(layer)*bb(layer)*at(layer)*sin(m(i)*zt(layer)) - 2*zt(&
      layer)*zb(layer)*m(i)**2*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*&
      zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(&
      layer)**3*m(i)**4)*cb(layer)*bt(layer)*ab(layer)*sin(m(i)*zb(&
      layer)) + 2*zt(layer)*zb(layer)*m(i)**2*1.0/(zb(layer)**3*m(i)**4&
      - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i&
      )**4 - zt(layer)**3*m(i)**4)*cb(layer)*bt(layer)*ab(layer)*sin(m(&
      i)*zt(layer)) + 2*zt(layer)*zb(layer)*m(i)**2*1.0/(zb(layer)**3*m&
      (i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(&
      layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bt(layer)*at(&
      layer)*sin(m(i)*zb(layer)) - 2*zt(layer)*zb(layer)*m(i)**2*1.0/(&
      zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(&
      layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bt(&
      layer)*at(layer)*sin(m(i)*zt(layer)) - 2*zt(layer)*zb(layer)*m(i)&
      **2*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4&
      + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(&
      layer)*bb(layer)*ab(layer)*sin(m(i)*zb(layer)) + 2*zt(layer)*zb(&
      layer)*m(i)**2*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)&
      **2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i&
      )**4)*ct(layer)*bb(layer)*ab(layer)*sin(m(i)*zt(layer)) + 2*zt(&
      layer)*zb(layer)*m(i)**2*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*&
      zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(&
      layer)**3*m(i)**4)*ct(layer)*bb(layer)*at(layer)*sin(m(i)*zb(&
      layer)) - 2*zt(layer)*zb(layer)*m(i)**2*1.0/(zb(layer)**3*m(i)**4&
      - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i&
      )**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(layer)*at(layer)*sin(m(&
      i)*zt(layer)) + 2*zt(layer)*zb(layer)*m(i)**2*1.0/(zb(layer)**3*m&
      (i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(&
      layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bt(layer)*ab(&
      layer)*sin(m(i)*zb(layer)) - 2*zt(layer)*zb(layer)*m(i)**2*1.0/(&
      zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(&
      layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bt(&
      layer)*ab(layer)*sin(m(i)*zt(layer)) + 2*zt(layer)*zb(layer)*m(i)&
      **3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4&
      + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(&
      layer)*bb(layer)*at(layer)*zb(layer)*cos(m(i)*zb(layer)) - 2*zt(&
      layer)*zb(layer)*m(i)**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*&
      zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(&
      layer)**3*m(i)**4)*cb(layer)*bb(layer)*at(layer)*zt(layer)*cos(m(&
      i)*zt(layer)) + 2*zt(layer)*zb(layer)*m(i)**3*1.0/(zb(layer)**3*m&
      (i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(&
      layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bt(layer)*ab(&
      layer)*zb(layer)*cos(m(i)*zb(layer)) - 2*zt(layer)*zb(layer)*m(i)&
      **3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4&
      + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(&
      layer)*bt(layer)*ab(layer)*zt(layer)*cos(m(i)*zt(layer)) - 2*zt(&
      layer)*zb(layer)*m(i)**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*&
      zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(&
      layer)**3*m(i)**4)*cb(layer)*bt(layer)*at(layer)*zb(layer)*cos(m(&
      i)*zb(layer)) + 2*zt(layer)*zb(layer)*m(i)**3*1.0/(zb(layer)**3*m&
      (i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(&
      layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bt(layer)*at(&
      layer)*zt(layer)*cos(m(i)*zt(layer)) + 2*zt(layer)*zb(layer)*m(i)&
      **3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4&
      + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(&
      layer)*bb(layer)*ab(layer)*zb(layer)*cos(m(i)*zb(layer)) - 2*zt(&
      layer)*zb(layer)*m(i)**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*&
      zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(&
      layer)**3*m(i)**4)*ct(layer)*bb(layer)*ab(layer)*zt(layer)*cos(m(&
      i)*zt(layer)) - 2*zt(layer)*zb(layer)*m(i)**3*1.0/(zb(layer)**3*m&
      (i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(&
      layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(layer)*at(&
      layer)*zb(layer)*cos(m(i)*zb(layer)) + 2*zt(layer)*zb(layer)*m(i)&
      **3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4&
      + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(&
      layer)*bb(layer)*at(layer)*zt(layer)*cos(m(i)*zt(layer)) - 2*zt(&
      layer)*zb(layer)*m(i)**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*&
      zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(&
      layer)**3*m(i)**4)*ct(layer)*bt(layer)*ab(layer)*zb(layer)*cos(m(&
      i)*zb(layer)) + 2*zt(layer)*zb(layer)*m(i)**3*1.0/(zb(layer)**3*m&
      (i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(&
      layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bt(layer)*ab(&
      layer)*zt(layer)*cos(m(i)*zt(layer)) + zt(layer)*zb(layer)**2*m(i&
      )**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4&
      + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(&
      layer)*bt(layer)*at(layer)*cos(m(i)*zb(layer)) - zt(layer)*zb(&
      layer)**2*m(i)**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(&
      layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)&
      **3*m(i)**4)*cb(layer)*bt(layer)*at(layer)*cos(m(i)*zt(layer)) +&
      zt(layer)*zb(layer)**2*m(i)**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(&
      layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 -&
      zt(layer)**3*m(i)**4)*ct(layer)*bb(layer)*at(layer)*cos(m(i)*zb(&
      layer)) - zt(layer)*zb(layer)**2*m(i)**3*1.0/(zb(layer)**3*m(i)**&
      4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m&
      (i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(layer)*at(layer)*cos(&
      m(i)*zt(layer)) + zt(layer)*zb(layer)**2*m(i)**3*1.0/(zb(layer)**&
      3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(&
      layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bt(layer)*ab(&
      layer)*cos(m(i)*zb(layer)) - zt(layer)*zb(layer)**2*m(i)**3*1.0/(&
      zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(&
      layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bt(&
      layer)*ab(layer)*cos(m(i)*zt(layer)) + 3*zt(layer)**2*m(i)**2*1.0&
      /(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(&
      layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bb(&
      layer)*ab(layer)*sin(m(i)*zb(layer)) - 3*zt(layer)**2*m(i)**2*1.0&
      /(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(&
      layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bb(&
      layer)*ab(layer)*sin(m(i)*zt(layer)) - zt(layer)**2*m(i)**2*1.0/(&
      zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(&
      layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bb(&
      layer)*at(layer)*sin(m(i)*zb(layer)) + zt(layer)**2*m(i)**2*1.0/(&
      zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(&
      layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bb(&
      layer)*at(layer)*sin(m(i)*zt(layer)) - zt(layer)**2*m(i)**2*1.0/(&
      zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(&
      layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bt(&
      layer)*ab(layer)*sin(m(i)*zb(layer)) + zt(layer)**2*m(i)**2*1.0/(&
      zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(&
      layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bt(&
      layer)*ab(layer)*sin(m(i)*zt(layer)) - zt(layer)**2*m(i)**2*1.0/(&
      zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(&
      layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(&
      layer)*ab(layer)*sin(m(i)*zb(layer)) + zt(layer)**2*m(i)**2*1.0/(&
      zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(&
      layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(&
      layer)*ab(layer)*sin(m(i)*zt(layer)) - 3*zt(layer)**2*m(i)**3*1.0&
      /(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(&
      layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bb(&
      layer)*ab(layer)*zb(layer)*cos(m(i)*zb(layer)) + 3*zt(layer)**2*m&
      (i)**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)&
      **4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb&
      (layer)*bb(layer)*ab(layer)*zt(layer)*cos(m(i)*zt(layer)) + zt(&
      layer)**2*m(i)**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(&
      layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)&
      **3*m(i)**4)*cb(layer)*bb(layer)*at(layer)*zb(layer)*cos(m(i)*zb(&
      layer)) - zt(layer)**2*m(i)**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(&
      layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 -&
      zt(layer)**3*m(i)**4)*cb(layer)*bb(layer)*at(layer)*zt(layer)*cos&
      (m(i)*zt(layer)) + zt(layer)**2*m(i)**3*1.0/(zb(layer)**3*m(i)**4&
      - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i&
      )**4 - zt(layer)**3*m(i)**4)*cb(layer)*bt(layer)*ab(layer)*zb(&
      layer)*cos(m(i)*zb(layer)) - zt(layer)**2*m(i)**3*1.0/(zb(layer)&
      **3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*&
      zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bt(layer)*ab(&
      layer)*zt(layer)*cos(m(i)*zt(layer)) + zt(layer)**2*m(i)**3*1.0/(&
      zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(&
      layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(layer)*bb(&
      layer)*ab(layer)*zb(layer)*cos(m(i)*zb(layer)) - zt(layer)**2*m(i&
      )**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4&
      + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(&
      layer)*bb(layer)*ab(layer)*zt(layer)*cos(m(i)*zt(layer)) - zt(&
      layer)**2*zb(layer)*m(i)**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(&
      layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 -&
      zt(layer)**3*m(i)**4)*cb(layer)*bb(layer)*at(layer)*cos(m(i)*zb(&
      layer)) + zt(layer)**2*zb(layer)*m(i)**3*1.0/(zb(layer)**3*m(i)**&
      4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m&
      (i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bb(layer)*at(layer)*cos(&
      m(i)*zt(layer)) - zt(layer)**2*zb(layer)*m(i)**3*1.0/(zb(layer)**&
      3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(layer)**2*zb(&
      layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bt(layer)*ab(&
      layer)*cos(m(i)*zb(layer)) + zt(layer)**2*zb(layer)*m(i)**3*1.0/(&
      zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4 + 3*zt(&
      layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*cb(layer)*bt(&
      layer)*ab(layer)*cos(m(i)*zt(layer)) - zt(layer)**2*zb(layer)*m(i&
      )**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)**2*m(i)**4&
      + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i)**4)*ct(&
      layer)*bb(layer)*ab(layer)*cos(m(i)*zb(layer)) + zt(layer)**2*zb(&
      layer)*m(i)**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(layer)&
      **2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)**3*m(i&
      )**4)*ct(layer)*bb(layer)*ab(layer)*cos(m(i)*zt(layer)) + zt(&
      layer)**3*m(i)**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(&
      layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)&
      **3*m(i)**4)*cb(layer)*bb(layer)*ab(layer)*cos(m(i)*zb(layer)) -&
      zt(layer)**3*m(i)**3*1.0/(zb(layer)**3*m(i)**4 - 3*zt(layer)*zb(&
      layer)**2*m(i)**4 + 3*zt(layer)**2*zb(layer)*m(i)**4 - zt(layer)&
      **3*m(i)**4)*cb(layer)*bb(layer)*ab(layer)*cos(m(i)*zt(layer)))
          END DO         
        END DO    
                    
      END SUBROUTINE
     

      SUBROUTINE dim1sin_D_aDb_linear(m, at, ab, bt, bb, zt, zb, a, &
                                      neig, nlayers) 
        USE types
        IMPLICIT NONE     
        
        INTEGER, intent(in) :: neig        
        INTEGER, intent(in) :: nlayers
        REAL(DP), intent(in), dimension(0:neig-1) ::m
        REAL(DP), intent(in), dimension(0:nlayers-1) :: at,ab,bt,bb,&
                                                        zt,zb
        REAL(DP), intent(out), dimension(0:neig-1) :: a
        INTEGER :: i, layer
        a=0.0D0
        DO layer = 0, nlayers-1
          DO i = 0, neig-1
      a(i) = a(i) + (-(zb(layer) - zt(layer))**(-2)*1.0/m(i)*(bb(layer)&
      - bt(layer))*(ab(layer) - at(layer))*cos(m(i)*zb(layer)) + (zb(&
      layer) - zt(layer))**(-2)*1.0/m(i)*(bb(layer) - bt(layer))*(ab(&
      layer) - at(layer))*cos(m(i)*zt(layer)) - 1.0/(zb(layer) - zt(&
      layer))*(bb(layer) - bt(layer))*(1.0/(zb(layer) - zt(layer))*(ab(&
      layer) - at(layer))*(zb(layer) - zt(layer)) + at(layer))*sin(m(i)&
      *zb(layer)) + 1.0/(zb(layer) - zt(layer))*(bb(layer) - bt(layer))&
      *at(layer)*sin(m(i)*zt(layer)))
          END DO         
        END DO    
        
        DO i = 0, neig-1
      a(i) = a(i) + (-1.0/(zb(0) - zt(0))*(bb(0) - bt(0))*at(0)*sin(m(i)&
      *zt(0)) + 1.0/(zb(nlayers - 1) - zt(nlayers - 1))*(bb(nlayers - 1&
      ) - bt(nlayers - 1))*(1.0/(zb(nlayers - 1) - zt(nlayers - 1))*(ab&
      (nlayers - 1) - at(nlayers - 1))*(zb(nlayers - 1) - zt(nlayers -&
      1)) + at(nlayers - 1))*sin(m(i)*zb(nlayers - 1)))
        END DO         
                    
      END SUBROUTINE

      
