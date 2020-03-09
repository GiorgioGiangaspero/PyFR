# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

// #include <stdio.h>

<% twicepi = 2.0*math.pi %>

<%pyfr:macro name='gaussian' params='csi, GC, sigma, output'>
    output = (1./sigma/sqrt(${twicepi}*GC))*exp(-0.5*(pow(csi/sigma,2)));
</%pyfr:macro>

<%pyfr:kernel name='negdivconf' ndim='2'
              t='scalar fpdtype_t'
              tdivtconf='inout fpdtype_t[${str(nvars)}]'
              ploc='in fpdtype_t[${str(ndims)}]'
              u='in fpdtype_t[${str(nvars)}]'
              rcpdjac='in fpdtype_t'>
% for i, ex in enumerate(srcex):
    tdivtconf[${i}] = -rcpdjac*tdivtconf[${i}] + ${ex};
% endfor

if (affected[0] > 0.0){
// Turbulent characteristic lengths (radii of influence)
fpdtype_t lturb[${ndims}][${ndims}] = ${lturb};

// Guassian constants
fpdtype_t GCs[${ndims}][${ndims}] = ${GCs};
fpdtype_t sigma = ${sigma};

// Initialize the turbsrc to 0.0
fpdtype_t turbsrc[${ndims}] = {0.0};

// Working variables
fpdtype_t g, csi, GC, output;

// Loop over the eddies
% for n in range(N):
    // n = ${n};
    // printf("Eddy: t=%f, eddies_loc=(%f, %f, %f), n=%d\n", t, eddies_loc[0][${n}], eddies_loc[1][${n}], eddies_loc[2][${n}], n);

    //U,V,W
    % for j in range(ndims):
        g = 1.0;

        csi = fabs((ploc[2] - eddies_loc[2][${n}])/lturb[2][${j}]);
        if (csi < 1.0){
            GC  = GCs[2][${j}];
            ${pyfr.expand('gaussian', 'csi', 'GC', 'sigma', 'output')};
            g *= output;

            csi = fabs((ploc[1] - eddies_loc[1][${n}])/lturb[1][${j}]);
            if (csi < 1.0){
                GC  = GCs[1][${j}];
                ${pyfr.expand('gaussian', 'csi', 'GC', 'sigma', 'output')};
                g *= output;

                csi = fabs((ploc[0] - eddies_loc[0][${n}])/lturb[0][${j}]);
                if (csi < 1.0){
                    GC  = GCs[0][${j}];
                    ${pyfr.expand('gaussian', 'csi', 'GC', 'sigma', 'output')};
                    g *= output;

                    // Accumulate taking into account this components strength
                    turbsrc[${j}] += g*eddies_strength[${j}][${n}];
               }
            }
        }
    % endfor
% endfor

// order is important here.
turbsrc[2] = aij[3]*turbsrc[2];
turbsrc[1] = aij[1]*turbsrc[0] + aij[2]*turbsrc[1];
turbsrc[0] = aij[0]*turbsrc[0];

// source term for synthetic turbulence, only for the momentum equations for the
// moment. Multiply by the density to make it dimensionally consistent for a
// compressible solver.
% for i in range(ndims):
    % if system == 'compr':
        tdivtconf[${i} + 1] += u[0]*factor[${i}]*turbsrc[${i}];
    % else:
        tdivtconf[${i} + 1] += factor[${i}]*turbsrc[${i}];
    % endif
% endfor


// TODO add pressure (i.e. energy) and density fluctuations for Ma > 0.3 flows,
// (compressible solver only, of course).
}
</%pyfr:kernel>
