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


