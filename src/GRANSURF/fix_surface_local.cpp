/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_surface_local.h"

#include "atom.h"
#include "atom_vec_line.h"
#include "atom_vec_tri.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "math_const.h"
#include "math_extra.h"
#include "memory.h"
#include "molecule.h"
#include "stl_reader.h"

#include <map>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

// NOTE: add input keyword option for multiple mol/STL files, like FSG
// NOTE: add optional flat keyword
// NOTE: add debug method here and in global to print out full Connect2d/3d for each line/tri
//       so can debug/compare between global and local
// NOTE: need one allocater for ints, one for tagints ?

// NOTE: change defines to static const here and in global
// NOTE: what if lines/tris already exist (from data file)
//       when read more from STL or molecule file ?
// NOTE: how to set and limit MAXTRIPOINT for allocators, no overallocation now
// NOTE: print out stats on connections, same as fix surf/global
// NOTE: need more tallying in memory_usage()
// NOTE: total bin count for Rvous seems too large when small # of surfs - check nbins ?
// NOTE: make DELTA values bigger when done testing
// NOTE: num in point_match() is just for debugging, can remove it
// NOTE: call to domain->remap() in assign2d/3d() will wrap new mol/STL line/tri particles by PBC
//       is that what we want ?   not the case for fix surf/global lines/tris

#define EPSILON 0.001
#define NBIN 100
#define BIG 1.0e20
#define MAXLINE 256
#define MAXTRIPOINT 24

#define DELTA 128
#define DELTA_CONNECT 4     // make it larger when done testing
#define DELTA_RVOUS 8       // must be >= 8, make it bigger when done testing

enum{FLAT,CONCAVE,CONVEX};
enum{SAME_SIDE,OPPOSITE_SIDE};

#define FLATTHRESH 1.0-cos(MY_PI/180.0)    // default = 1 degree

static constexpr int RVOUS = 1;   // 0 for irregular, 1 for all2all

enum{DATAFILE,MOLTEMPLATE,STLFILE};
enum{LAYOUT_UNIFORM,LAYOUT_NONUNIFORM,LAYOUT_TILED};    // several files

// allocate space for static class variable

FixSurfaceLocal *FixSurfaceLocal::fptr;

/* ---------------------------------------------------------------------- */

FixSurfaceLocal::FixSurfaceLocal(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg != 4) error->all(FLERR,"Illegal fix surface/local command");

  create_attribute = 1;

  dimension = domain->dimension;

  atom2connect = nullptr;
  grow_arrays(atom->nmax);
  atom->add_callback(0);
  atom->add_callback(2);

  nlocal_connect = nghost_connect = nmax_connect = 0;
  connect2d = nullptr;
  connect3d = nullptr;
  pool2d = nullptr;
  pool3d = nullptr;
  connect2atom = nullptr;
  
  tcp = new MyPoolChunk<tagint>(1,MAXTRIPOINT,6);

  // 3 possible sources of lines/tris
  // (1) mode = DATAFILE, lines/tris were already read from data file
  // (2) mode = MOLTEMPLATE, lines/tris are in molecule template ID
  // {3} mode = STLFILE, tris are in an STL file

  sourceID = nullptr;

  if (strcmp(arg[3],"NULL") == 0) mode = DATAFILE;
  else {
    int n = strlen(arg[3]) + 1;
    sourceID = new char[n];
    strcpy(sourceID,arg[3]);
    int imol = atom->find_molecule(sourceID);
    if (imol >= 0) mode = MOLTEMPLATE;
    else mode = STLFILE;
  }

  flatthresh = FLATTHRESH;
  
  flag_complete = 0;
  epssq = -1.0;
}

/* ---------------------------------------------------------------------- */

FixSurfaceLocal::~FixSurfaceLocal()
{
  // unregister callbacks to this fix from Atom class

  atom->delete_callback(id,0);

  delete [] sourceID;
  memory->destroy(atom2connect);
  memory->destroy(connect2atom);

  // return all connection vector memory to TCP

  if (dimension == 2) {
    int nall = nlocal_connect + nghost_connect;
    for (int i = 0; i < nall; i++) {
      if (connect2d[i].neigh_p1) {
        tcp->put(pool2d[i].neigh_p1);
        tcp->put(pool2d[i].pwhich_p1);
        tcp->put(pool2d[i].nside_p1);
        tcp->put(pool2d[i].aflag_p1);
      }
      if (connect2d[i].neigh_p2) {
        tcp->put(pool2d[i].neigh_p2);
        tcp->put(pool2d[i].pwhich_p2);
        tcp->put(pool2d[i].nside_p2);
        tcp->put(pool2d[i].aflag_p2);
      }
    }
    memory->sfree(connect2d);
  } else {
    int nall = nlocal_connect + nghost_connect;
    for (int i = 0; i < nall; i++) {
      if (connect3d[i].neigh_e1) {
        tcp->put(pool3d[i].neigh_e1);
        tcp->put(pool3d[i].ewhich_e1);
        tcp->put(pool3d[i].nside_e1);
        tcp->put(pool3d[i].aflag_e1);
      }
      if (connect3d[i].neigh_e2) {
        tcp->put(pool3d[i].neigh_e2);
        tcp->put(pool3d[i].ewhich_e2);
        tcp->put(pool3d[i].nside_e2);
        tcp->put(pool3d[i].aflag_e2);
      }
      if (connect3d[i].neigh_e3) {
        tcp->put(pool3d[i].neigh_e3);
        tcp->put(pool3d[i].ewhich_e3);
        tcp->put(pool3d[i].nside_e3);
        tcp->put(pool3d[i].aflag_e3);
      }
      if (connect3d[i].neigh_c1) {
        tcp->put(pool3d[i].neigh_c1);
        tcp->put(pool3d[i].cwhich_c1);
      }
      if (connect3d[i].neigh_c2) {
        tcp->put(pool3d[i].neigh_c2);
        tcp->put(pool3d[i].cwhich_c2);
      }
      if (connect3d[i].neigh_c3) {
        tcp->put(pool3d[i].neigh_c3);
        tcp->put(pool3d[i].cwhich_c3);
      }
    }
    memory->sfree(connect3d);
  }

  memory->sfree(pool2d);
  memory->sfree(pool3d);

  delete tcp;
}

/* ---------------------------------------------------------------------- */

int FixSurfaceLocal::setmask()
{
  int mask = 0;
  mask |= PRE_NEIGHBOR;    // only needed for DEBUG tests
  return mask;
}

/* ----------------------------------------------------------------------
   one-time setup of distributed lines/tri and their connectivity
   must be done in post_constructor() for MOLTEMPLATE and STLFILE
     they add owned lines/tris to AtomVec class
     its grow() method makes a callback to grow_arrays() in this fix
     callback can't be invoked unless fix is fully instantiated
------------------------------------------------------------------------- */

void FixSurfaceLocal::post_constructor()
{
  // for mode == DATAFILE
  // initialize connectivity of already existing lines/tris

  if (mode == DATAFILE) {
    if (comm->me == 0 && screen)
      fprintf(screen,"Connecting line/tri particles ...\n");

    if (dimension == 2) connectivity2d_local();
    else connectivity3d_local();
  }

  if (mode == MOLTEMPLATE || mode == STLFILE) {

    // error check that no line/tri particles already exist
    //   b/c no connectivity would be produced for them

    if (check_exist())
      error->all(FLERR,"Fix surface/local with non-NULL source when lines/tris already exist");

    if (comm->me == 0 && screen) {
      if (mode == MOLTEMPLATE)
        fprintf(screen,"Converting molecule file to line/tri particles ...\n");
      if (mode == STLFILE)
        fprintf(screen,"Reading STL file for triangle particles ...\n");
    }

    points = nullptr;
    lines = nullptr;
    tris = nullptr;

    // read in lines/tris from appropriate sourrce
    // each proc builds global data structs of points/lines/tris

    if (mode == MOLTEMPLATE) extract_from_molecules(sourceID);
    if (mode == STLFILE) extract_from_stlfile(sourceID);

    // each proc infers global connectivity from global data structs
    // distribute lines/surfs across procs, based on center pt coords

    if (dimension == 2) {
      connectivity2d_global();
      assign2d();
    } else {
      connectivity3d_global();
      assign3d();
    }

    bigint nblocal = atom->nlocal;
    MPI_Allreduce(&nblocal, &atom->natoms, 1, MPI_LMP_BIGINT, MPI_SUM, world);

    // delete global data structs

    memory->sfree(points);
    memory->sfree(lines);
    memory->sfree(tris);
  }

  // set max size for comm of connection info
  // 2d = 2 end points, 4 vectors, each of length npmaxall
  // 3d = 3 edges, 4 vectors, each of length nemaxall
  //      plus 3 corner points, 2 vectors, each of length ncmaxall
  
  if (dimension == 2) {
    int nlocal = atom->nlocal;
    int *line = atom->line;
    int iconnect;
    
    int npmax = 0;
    for (int i = 0; i < nlocal; i++) {
      if (line[i] < 0) continue;
      iconnect = atom2connect[i];
      npmax = MAX(npmax,connect2d[iconnect].np1);
      npmax = MAX(npmax,connect2d[iconnect].np2);
    }

    int npmaxall;
    MPI_Allreduce(&npmax,&npmaxall,1,MPI_INT,MPI_MAX,world);
    comm_border = 3 + 2*4*npmaxall;
    
  } else if (dimension== 3) { 
    int nlocal = atom->nlocal;
    int *tri = atom->tri;
    int iconnect;
    
    int nemax = 0;
    int ncmax = 0;
    for (int i = 0; i < nlocal; i++) {
      if (tri[i] < 0) continue;
      iconnect = atom2connect[i];
      nemax = MAX(nemax,connect3d[iconnect].ne1);
      nemax = MAX(nemax,connect3d[iconnect].ne2);
      nemax = MAX(nemax,connect3d[iconnect].ne3);
      ncmax = MAX(ncmax,connect3d[iconnect].nc1);
      ncmax = MAX(ncmax,connect3d[iconnect].nc2);
      ncmax = MAX(ncmax,connect3d[iconnect].nc3);
    }

    int nemaxall,ncmaxall;
    MPI_Allreduce(&nemax,&nemaxall,1,MPI_INT,MPI_MAX,world);
    MPI_Allreduce(&ncmax,&ncmaxall,1,MPI_INT,MPI_MAX,world);
    comm_border = 4 + 3*4*nemaxall + 3*2*ncmaxall;
  }

  // output stats, same as FSG
  
}

/* ----------------------------------------------------------------------
   allocate atom-based array
------------------------------------------------------------------------- */

void FixSurfaceLocal::grow_arrays(int nmax)
{
  memory->grow(atom2connect,nmax,"surface/local:atom2connect");
}

/* ---------------------------------------------------------------------- */

void FixSurfaceLocal::setup_pre_neighbor()
{
  // one-time calcualtion of remaining fields in Connect2d/3d
  // cannot do until now, b/c need ghost connection info via border comm
  // re-communicate new owned-particle connectivity to ghost particles
  
  if (!flag_complete) {
    if (dimension == 2) connectivity2d_complete();
    else connectivity3d_complete();
    flag_complete = 1;

    clear_bonus();
    comm->forward_comm(this);
  }

  pre_neighbor();
}

/* ----------------------------------------------------------------------
   only for DEBUGGING
   check that atom2connect is identical to line/tri
   check that atom2connect is consistent with connect2atom
------------------------------------------------------------------------- */

void FixSurfaceLocal::pre_neighbor()
{
  int n = atom->nlocal + atom->nghost;

  int *surf = atom->line;
  if (dimension == 3) surf = atom->tri;

  int count1 = 0;
  int count2 = 0;

  for (int i = 0; i < n; i++) {
    if (surf[i] < 0) continue;

    if (atom2connect[i] != surf[i]) {
      count1++;
      continue;
    }
    if (connect2atom[atom2connect[i]] != i) count2++;
  }

  int all1,all2;
  MPI_Allreduce(&count1,&all1,1,MPI_INT,MPI_SUM,world);
  MPI_Allreduce(&count2,&all2,1,MPI_INT,MPI_SUM,world);

  if (all1 || all2 && comm->me == 0) {
    char str[128];
    sprintf(str,"FSL atom2connect vector mis-match: %d %d\n",all1,all2);
    error->warning(FLERR,str);
  }
}

/* ----------------------------------------------------------------------
   grow connect array by DELTA_CONNECT
------------------------------------------------------------------------- */

void FixSurfaceLocal::grow_connect()
{
  nmax_connect = nmax_connect/DELTA_CONNECT * DELTA_CONNECT;
  bigint newmax = (bigint) nmax_connect + DELTA_CONNECT;
  if (newmax > MAXSMALLINT) error->one(FLERR,"Too many fix surface/local connections");
  nmax_connect = newmax;

  memory->grow(connect2atom,nmax_connect,"surface/local:connect2atom");

  if (dimension == 2) {
    connect2d = (Connect2d *)
      memory->srealloc(connect2d,nmax_connect*sizeof(Connect2d),
                       "surface/local:connect2d");
    pool2d = (Pool2d *)
      memory->srealloc(pool2d,nmax_connect*sizeof(Pool2d),
                       "surface/local:pool2d");
  } else {
    connect3d = (Connect3d *)
      memory->srealloc(connect3d,nmax_connect*sizeof(Connect3d),
                       "surface/local:connect3d");
    pool3d = (Pool3d *)
      memory->srealloc(pool3d,nmax_connect*sizeof(Pool3d),
                       "surface/local:pool3d");
  }
}

/* ----------------------------------------------------------------------
   copy atom I to atom J with optional delflag
   I or J or both can be a line/triangle with connection info
------------------------------------------------------------------------- */

void FixSurfaceLocal::copy_arrays(int i, int j, int delflag)
{
  // if overwriting atom J via delflag and J is a line/tri with connection data:
  //   return J's vector memory to TCP before memcpy()
  //   shrink list of owned connections by copying last element to Kth location

  if (dimension == 2) {
    if (delflag && atom2connect[j] >= 0) {
      int k = atom2connect[j];
      if (connect2d[k].neigh_p1) {
        tcp->put(pool2d[k].neigh_p1);
        tcp->put(pool2d[k].pwhich_p1);
        tcp->put(pool2d[k].nside_p1);
        tcp->put(pool2d[k].aflag_p1);
      }
      if (connect2d[k].neigh_p2) {
        tcp->put(pool2d[k].neigh_p2);
        tcp->put(pool2d[k].pwhich_p2);
        tcp->put(pool2d[k].nside_p2);
        tcp->put(pool2d[k].aflag_p2);
      }
      atom2connect[connect2atom[nlocal_connect-1]] = k;
      memcpy(&connect2d[k],&connect2d[nlocal_connect-1],sizeof(Connect2d));
      memcpy(&pool2d[k],&pool2d[nlocal_connect-1],sizeof(Pool2d));
      nlocal_connect--;
    }
  } else {
    if (delflag && atom2connect[j] >= 0) {
      int k = atom2connect[j];
      if (connect3d[k].neigh_e1) {
        tcp->put(pool3d[k].neigh_e1);
        tcp->put(pool3d[k].ewhich_e1);
        tcp->put(pool3d[k].nside_e1);
        tcp->put(pool3d[k].aflag_e1);
      }
      if (connect3d[k].neigh_e2) {
        tcp->put(pool3d[k].neigh_e2);
        tcp->put(pool3d[k].ewhich_e2);
        tcp->put(pool3d[k].nside_e2);
        tcp->put(pool3d[k].aflag_e2);
      }
      if (connect3d[k].neigh_e3) {
        tcp->put(pool3d[k].neigh_e3);
        tcp->put(pool3d[k].ewhich_e3);
        tcp->put(pool3d[k].nside_e3);
        tcp->put(pool3d[k].aflag_e3);
      }
      if (connect3d[k].neigh_c1) {
        tcp->put(pool3d[k].neigh_c1);
        tcp->put(pool3d[k].cwhich_c1);
      }
      if (connect3d[k].neigh_c2) {
        tcp->put(pool3d[k].neigh_c2);
        tcp->put(pool3d[k].cwhich_c2);
      }
      if (connect3d[k].neigh_c3) {
        tcp->put(pool3d[k].neigh_c3);
        tcp->put(pool3d[k].cwhich_c3);
      }
      atom2connect[connect2atom[nlocal_connect-1]] = k;
      memcpy(&connect3d[k],&connect3d[nlocal_connect-1],sizeof(Connect3d));
      nlocal_connect--;
    }
  }

  // if atom I has connection data, reset connect2atom[I] to loc J
  // do NOT do this if self-copy (I=J) since I's connection data is already deleted

  if (atom2connect[i] >= 0 && i != j) connect2atom[atom2connect[i]] = j;
  atom2connect[j] = atom2connect[i];
}

/* ----------------------------------------------------------------------
   initialize one atom's array values, called when atom is created
------------------------------------------------------------------------- */

void FixSurfaceLocal::set_arrays(int i)
{
  atom2connect[i] = -1;
}

/* ----------------------------------------------------------------------
   clear ghost info in connect array
   called before ghosts are recommunicated in comm and irregular
   return all vector memory to TCP
------------------------------------------------------------------------- */

void FixSurfaceLocal::clear_bonus()
{
  if (dimension == 2) {
    int nall = nlocal_connect + nghost_connect;
    for (int i = nlocal_connect; i < nall; i++) {
      if (connect2d[i].neigh_p1) {
        tcp->put(pool2d[i].neigh_p1);
        tcp->put(pool2d[i].pwhich_p1);
        tcp->put(pool2d[i].nside_p1);
        tcp->put(pool2d[i].aflag_p1);
      }
      if (connect2d[i].neigh_p2) {
        tcp->put(pool2d[i].neigh_p2);
        tcp->put(pool2d[i].pwhich_p2);
        tcp->put(pool2d[i].nside_p2);
        tcp->put(pool2d[i].aflag_p2);
      }
    }
  } else if (dimension == 3) {
    int nall = nlocal_connect + nghost_connect;
    for (int i = nlocal_connect; i < nall; i++) {
      if (connect3d[i].neigh_e1) {
        tcp->put(pool3d[i].neigh_e1);
        tcp->put(pool3d[i].ewhich_e1);
        tcp->put(pool3d[i].nside_e1);
        tcp->put(pool3d[i].aflag_e1);
      }
      if (connect3d[i].neigh_e2) {
        tcp->put(pool3d[i].neigh_e2);
        tcp->put(pool3d[i].ewhich_e2);
        tcp->put(pool3d[i].nside_e2);
        tcp->put(pool3d[i].aflag_e2);
      }
      if (connect3d[i].neigh_e3) {
        tcp->put(pool3d[i].neigh_e3);
        tcp->put(pool3d[i].ewhich_e3);
        tcp->put(pool3d[i].nside_e3);
        tcp->put(pool3d[i].aflag_e3);
      }
      if (connect3d[i].neigh_c1) {
        tcp->put(pool3d[i].neigh_c1);
        tcp->put(pool3d[i].cwhich_c1);
      }
      if (connect3d[i].neigh_c2) {
        tcp->put(pool3d[i].neigh_c2);
        tcp->put(pool3d[i].cwhich_c2);
      }
      if (connect3d[i].neigh_c3) {
        tcp->put(pool3d[i].neigh_c3);
        tcp->put(pool3d[i].cwhich_c3);
      }
    }
  }

  nghost_connect = 0;
}

/* ----------------------------------------------------------------------
   pack values for border communication at re-neighboring
------------------------------------------------------------------------- */

int FixSurfaceLocal::pack_border(int n, int *list, double *buf)
{
  int i,j,k,m,ic,nc;

  m = 0;

  if (dimension == 2) {
    int np1,np2;

    for (i = 0; i < n; i++) {
      j = list[i];
      if (atom2connect[j] < 0) buf[m++] = ubuf(0).d;
      else {
        buf[m++] = ubuf(1).d;
        ic = atom2connect[j];

        np1 = connect2d[ic].np1;
        np2 = connect2d[ic].np2;
        buf[m++] = ubuf(np1).d;
        buf[m++] = ubuf(np2).d;
        if (np1)
          for (k = 0; k < np1; k++) {
            buf[m++] = ubuf(connect2d[ic].neigh_p1[k]).d;
            buf[m++] = ubuf(connect2d[ic].pwhich_p1[k]).d;
            buf[m++] = ubuf(connect2d[ic].nside_p1[k]).d;
            buf[m++] = ubuf(connect2d[ic].aflag_p1[k]).d;
          }
        if (np2)
          for (k = 0; k < np2; k++) {
            buf[m++] = ubuf(connect2d[ic].neigh_p2[k]).d;
            buf[m++] = ubuf(connect2d[ic].pwhich_p2[k]).d;
            buf[m++] = ubuf(connect2d[ic].nside_p2[k]).d;
            buf[m++] = ubuf(connect2d[ic].aflag_p2[k]).d;
          }
      }
    }

  } else {
    int ne1,ne2,ne3,nc1,nc2,nc3;

    for (i = 0; i < n; i++) {
      j = list[i];
      if (atom2connect[j] < 0) buf[m++] = ubuf(0).d;
      else {
        buf[m++] = ubuf(1).d;
        ic = atom2connect[j];

        ne1 = connect3d[ic].ne1;
        ne2 = connect3d[ic].ne2;
        ne3 = connect3d[ic].ne3;
        buf[m++] = ubuf(ne1).d;
        buf[m++] = ubuf(ne2).d;
        buf[m++] = ubuf(ne3).d;
        if (ne1)
          for (k = 0; k < ne1; k++) {
            buf[m++] = ubuf(connect3d[ic].neigh_e1[k]).d;
            buf[m++] = ubuf(connect3d[ic].ewhich_e1[k]).d;
            buf[m++] = ubuf(connect3d[ic].nside_e1[k]).d;
            buf[m++] = ubuf(connect3d[ic].aflag_e1[k]).d;
          }
        if (ne2)
          for (k = 0; k < ne2; k++) {
            buf[m++] = ubuf(connect3d[ic].neigh_e2[k]).d;
            buf[m++] = ubuf(connect3d[ic].ewhich_e2[k]).d;
            buf[m++] = ubuf(connect3d[ic].nside_e2[k]).d;
            buf[m++] = ubuf(connect3d[ic].aflag_e2[k]).d;
          }
        }
        if (ne3)
          for (k = 0; k < ne3; k++) {
            buf[m++] = ubuf(connect3d[ic].neigh_e3[k]).d;
            buf[m++] = ubuf(connect3d[ic].ewhich_e3[k]).d;
            buf[m++] = ubuf(connect3d[ic].nside_e3[k]).d;
            buf[m++] = ubuf(connect3d[ic].aflag_e3[k]).d;
          }

        nc1 = connect3d[ic].nc1;
        nc2 = connect3d[ic].nc2;
        nc3 = connect3d[ic].nc3;
        buf[m++] = ubuf(nc1).d;
        buf[m++] = ubuf(nc2).d;
        buf[m++] = ubuf(nc3).d;
        if (nc1)
          for (k = 0; k < nc1; k++) {
            buf[m++] = ubuf(connect3d[ic].neigh_c1[k]).d;
            buf[m++] = ubuf(connect3d[ic].cwhich_c1[k]).d;
          }
        if (nc2)
          for (k = 0; k < nc2; k++) {
            buf[m++] = ubuf(connect3d[ic].neigh_c2[k]).d;
            buf[m++] = ubuf(connect3d[ic].cwhich_c2[k]).d;
          }
        if (nc3)
          for (k = 0; k < nc3; k++) {
            buf[m++] = ubuf(connect3d[ic].neigh_c3[k]).d;
            buf[m++] = ubuf(connect3d[ic].cwhich_c3[k]).d;
          }
    }
  }

  return m;
}

/* ----------------------------------------------------------------------
   unpack values for border communication at re-neighboring
------------------------------------------------------------------------- */

int FixSurfaceLocal::unpack_border(int n, int first, double *buf)
{
  int i,j,k,m,last,flag;

  m = 0;
  last = first + n;

  if (dimension == 2) {
    int np1,np2;

    for (i = first; i < last; i++) {
      flag = (int) ubuf(buf[m++]).i;
      if (flag == 0) atom2connect[i] = -1;
      else {
        j = nlocal_connect + nghost_connect;
        if (j == nmax_connect) grow_connect();

        np1 = (int) ubuf(buf[m++]).i;
        np2 = (int) ubuf(buf[m++]).i;
        connect2d[j].np1 = np1;
        connect2d[j].np2 = np2;
        
        if (np1) {
          connect2d[j].neigh_p1 = tcp->get(np1,pool2d[j].neigh_p1);
          connect2d[j].pwhich_p1 = tcp->get(np1,pool2d[j].pwhich_p1);
          connect2d[j].nside_p1 = tcp->get(np1,pool2d[j].nside_p1);
          connect2d[j].aflag_p1 = tcp->get(np1,pool2d[j].aflag_p1);
          for (k = 0; k < np1; k++) {
            connect2d[j].neigh_p1[k] = (tagint) ubuf(buf[m++]).i;
            connect2d[j].pwhich_p1[k] = (tagint) ubuf(buf[m++]).i;
            connect2d[j].nside_p1[k] = (tagint) ubuf(buf[m++]).i;
            connect2d[j].aflag_p1[k] = (tagint) ubuf(buf[m++]).i;
          }
        } else {
          connect2d[j].neigh_p1 = nullptr;
          connect2d[j].pwhich_p1 = nullptr;
          connect2d[j].nside_p1 = nullptr;
          connect2d[j].aflag_p1 = nullptr;
        }
        
        if (np2) {
          connect2d[j].neigh_p2 = tcp->get(np1,pool2d[j].neigh_p2);
          connect2d[j].pwhich_p2 = tcp->get(np1,pool2d[j].pwhich_p2);
          connect2d[j].nside_p2 = tcp->get(np1,pool2d[j].nside_p2);
          connect2d[j].aflag_p2 = tcp->get(np1,pool2d[j].aflag_p2);
          for (k = 0; k < np1; k++) {
            connect2d[j].neigh_p2[k] = (tagint) ubuf(buf[m++]).i;
            connect2d[j].pwhich_p2[k] = (tagint) ubuf(buf[m++]).i;
            connect2d[j].nside_p2[k] = (tagint) ubuf(buf[m++]).i;
            connect2d[j].aflag_p2[k] = (tagint) ubuf(buf[m++]).i;
          }
        } else {
          connect2d[j].neigh_p2 = nullptr;
          connect2d[j].pwhich_p2 = nullptr;
          connect2d[j].nside_p2 = nullptr;
          connect2d[j].aflag_p2 = nullptr;
        }

        connect2atom[j] = i;
        atom2connect[i] = j;
        nghost_connect++;
      }
    }

  } else {
    int ne1,ne2,ne3,nc1,nc2,nc3;

    for (i = first; i < last; i++) {
      flag = (int) ubuf(buf[m++]).i;
      if (flag == 0) atom2connect[i] = -1;
      else {
        j = nlocal_connect + nghost_connect;
        if (j == nmax_connect) grow_connect();

        ne1 = (int) ubuf(buf[m++]).i;
        ne2 = (int) ubuf(buf[m++]).i;
        ne3 = (int) ubuf(buf[m++]).i;
        connect3d[j].ne1 = ne1;
        connect3d[j].ne2 = ne2;
        connect3d[j].ne3 = ne3;
        
        if (ne1) {
          connect3d[j].neigh_e1 = tcp->get(ne1,pool3d[j].neigh_e1);
          connect3d[j].ewhich_e1 = tcp->get(ne1,pool3d[j].ewhich_e1);
          connect3d[j].nside_e1 = tcp->get(ne1,pool3d[j].nside_e1);
          connect3d[j].aflag_e1 = tcp->get(ne1,pool3d[j].aflag_e1);
          for (k = 0; k < ne1; k++) {
            connect3d[j].neigh_e1[k] = (tagint) ubuf(buf[m++]).i;
            connect3d[j].ewhich_e1[k] = (tagint) ubuf(buf[m++]).i;
            connect3d[j].nside_e1[k] = (tagint) ubuf(buf[m++]).i;
            connect3d[j].aflag_e1[k] = (tagint) ubuf(buf[m++]).i;
          }
        } else {
          connect3d[j].neigh_e1 = nullptr;
          connect3d[j].ewhich_e1 = nullptr;
          connect3d[j].nside_e1 = nullptr;
          connect3d[j].aflag_e1 = nullptr;
        }

        if (ne2) {
          connect3d[j].neigh_e2 = tcp->get(ne2,pool3d[j].neigh_e2);
          connect3d[j].ewhich_e2 = tcp->get(ne2,pool3d[j].ewhich_e2);
          connect3d[j].nside_e2 = tcp->get(ne2,pool3d[j].nside_e2);
          connect3d[j].aflag_e2 = tcp->get(ne2,pool3d[j].aflag_e2);
          for (k = 0; k < ne2; k++) {
            connect3d[j].neigh_e2[k] = (tagint) ubuf(buf[m++]).i;
            connect3d[j].ewhich_e2[k] = (tagint) ubuf(buf[m++]).i;
            connect3d[j].nside_e2[k] = (tagint) ubuf(buf[m++]).i;
            connect3d[j].aflag_e2[k] = (tagint) ubuf(buf[m++]).i;
          }
        } else {
          connect3d[j].neigh_e2 = nullptr;
          connect3d[j].ewhich_e2 = nullptr;
          connect3d[j].nside_e2 = nullptr;
          connect3d[j].aflag_e2 = nullptr;
        }
        
        if (ne3) {
          connect3d[j].neigh_e3 = tcp->get(ne3,pool3d[j].neigh_e3);
          connect3d[j].ewhich_e3 = tcp->get(ne3,pool3d[j].ewhich_e3);
          connect3d[j].nside_e3 = tcp->get(ne3,pool3d[j].nside_e3);
          connect3d[j].aflag_e3 = tcp->get(ne3,pool3d[j].aflag_e3);
          for (k = 0; k < ne3; k++) {
            connect3d[j].neigh_e3[k] = (tagint) ubuf(buf[m++]).i;
            connect3d[j].ewhich_e3[k] = (tagint) ubuf(buf[m++]).i;
            connect3d[j].nside_e3[k] = (tagint) ubuf(buf[m++]).i;
            connect3d[j].aflag_e3[k] = (tagint) ubuf(buf[m++]).i;
          }
        } else {
          connect3d[j].neigh_e3 = nullptr;
          connect3d[j].ewhich_e3 = nullptr;
          connect3d[j].nside_e3 = nullptr;
          connect3d[j].aflag_e3 = nullptr;
        }

        nc1 = (int) ubuf(buf[m++]).i;
        nc2 = (int) ubuf(buf[m++]).i;
        nc3 = (int) ubuf(buf[m++]).i;
        connect3d[j].nc1 = nc1;
        connect3d[j].nc2 = nc2;
        connect3d[j].nc3 = nc3;
        
        if (nc1) {
          connect3d[j].neigh_c1 = tcp->get(nc1,pool3d[j].neigh_c1);
          connect3d[j].cwhich_c1 = tcp->get(nc1,pool3d[j].cwhich_c1);
          for (k = 0; k < nc1; k++) {
            connect3d[j].neigh_c1[k] = (tagint) ubuf(buf[m++]).i;
            connect3d[j].cwhich_c1[k] = (tagint) ubuf(buf[m++]).i;
          }
        } else {
          connect3d[j].neigh_c1 = nullptr;
          connect3d[j].cwhich_c1 = nullptr;
        }
        
        if (nc2) {
          connect3d[j].neigh_c2 = tcp->get(nc2,pool3d[j].neigh_c2);
          connect3d[j].cwhich_c2 = tcp->get(nc2,pool3d[j].cwhich_c2);
          for (k = 0; k < nc1; k++) {
            connect3d[j].neigh_c2[k] = (tagint) ubuf(buf[m++]).i;
            connect3d[j].cwhich_c2[k] = (tagint) ubuf(buf[m++]).i;
          }
        } else {
          connect3d[j].neigh_c2 = nullptr;
          connect3d[j].cwhich_c2 = nullptr;
        }
        
        if (nc3) {
          connect3d[j].neigh_c3 = tcp->get(nc3,pool3d[j].neigh_c3);
          connect3d[j].cwhich_c3 = tcp->get(nc3,pool3d[j].cwhich_c3);
          for (k = 0; k < nc3; k++) {
            connect3d[j].neigh_c3[k] = (tagint) ubuf(buf[m++]).i;
            connect3d[j].cwhich_c3[k] = (tagint) ubuf(buf[m++]).i;
          }
        } else {
          connect3d[j].neigh_c3 = nullptr;
          connect3d[j].cwhich_c3 = nullptr;
        }
        
        connect2atom[j] = i;
        atom2connect[i] = j;
        nghost_connect++;
      }
    }
  }

  return m;
}

/* ----------------------------------------------------------------------
   pack values in local connect array for exchange with another proc
------------------------------------------------------------------------- */

int FixSurfaceLocal::pack_exchange(int i, double *buf)
{
  int j,k,n;
  int m = 0;

  if (dimension == 2) {
    int np1,np2;

    if (atom2connect[i] < 0) buf[m++] = ubuf(0).d;
    else {
      buf[m++] = ubuf(1).d;
      int j = atom2connect[i];

      np1 = connect2d[j].np1;
      np2 = connect2d[j].np2;
      buf[m++] = ubuf(np1).d;
      buf[m++] = ubuf(np2).d;
      if (np1)
        for (k = 0; k < np1; k++) {
          buf[m++] = ubuf(connect2d[j].neigh_p1[k]).d;
          buf[m++] = ubuf(connect2d[j].pwhich_p1[k]).d;
          buf[m++] = ubuf(connect2d[j].nside_p1[k]).d;
          buf[m++] = ubuf(connect2d[j].aflag_p1[k]).d;
        }
      if (np2)
        for (k = 0; k < np2; k++) {
          buf[m++] = ubuf(connect2d[j].neigh_p2[k]).d;
          buf[m++] = ubuf(connect2d[j].pwhich_p2[k]).d;
          buf[m++] = ubuf(connect2d[j].nside_p2[k]).d;
          buf[m++] = ubuf(connect2d[j].aflag_p2[k]).d;
        }
    }
      
  } else {
    int ne1,ne2,ne3,nc1,nc2,nc3;

    if (atom2connect[i] < 0) buf[m++] = ubuf(0).d;
    else {
      buf[m++] = ubuf(1).d;
      int j = atom2connect[i];

      ne1 = connect3d[j].ne1;
      ne2 = connect3d[j].ne2;
      ne3 = connect3d[j].ne3;
      buf[m++] = ubuf(ne1).d;
      buf[m++] = ubuf(ne2).d;
      buf[m++] = ubuf(ne3).d;
      if (ne1)
        for (k = 0; k < ne1; k++) {
          buf[m++] = ubuf(connect3d[j].neigh_e1[k]).d;
          buf[m++] = ubuf(connect3d[j].ewhich_e1[k]).d;
          buf[m++] = ubuf(connect3d[j].nside_e1[k]).d;
          buf[m++] = ubuf(connect3d[j].aflag_e1[k]).d;
        }
      if (ne2)
        for (k = 0; k < ne2; k++) {
          buf[m++] = ubuf(connect3d[j].neigh_e2[k]).d;
          buf[m++] = ubuf(connect3d[j].ewhich_e2[k]).d;
          buf[m++] = ubuf(connect3d[j].nside_e2[k]).d;
          buf[m++] = ubuf(connect3d[j].aflag_e2[k]).d;
        }
      if (ne3)
        for (k = 0; k < ne3; k++) {
          buf[m++] = ubuf(connect3d[j].neigh_e3[k]).d;
          buf[m++] = ubuf(connect3d[j].ewhich_e3[k]).d;
          buf[m++] = ubuf(connect3d[j].nside_e3[k]).d;
          buf[m++] = ubuf(connect3d[j].aflag_e3[k]).d;
        }

      nc1 = connect3d[j].nc1;
      nc2 = connect3d[j].nc2;
      nc3 = connect3d[j].nc3;
      buf[m++] = ubuf(nc1).d;
      buf[m++] = ubuf(nc2).d;
      buf[m++] = ubuf(nc3).d;
      if (nc1)
        for (k = 0; k < nc1; k++) {
          buf[m++] = ubuf(connect3d[j].neigh_c1[k]).d;
          buf[m++] = ubuf(connect3d[j].cwhich_c1[k]).d;
        }
      if (nc2)
        for (k = 0; k < nc2; k++) {
          buf[m++] = ubuf(connect3d[j].neigh_c2[k]).d;
          buf[m++] = ubuf(connect3d[j].cwhich_c2[k]).d;
        }
      if (nc3)
        for (k = 0; k < nc3; k++) {
          buf[m++] = ubuf(connect3d[j].neigh_c3[k]).d;
          buf[m++] = ubuf(connect3d[j].cwhich_c3[k]).d;
        }
    }
  }

  return m;
}

/* ----------------------------------------------------------------------
   unpack values in local connect array from exchange with another proc
------------------------------------------------------------------------- */

int FixSurfaceLocal::unpack_exchange(int nlocal, double *buf)
{
  int j,k,n,flag;
  int m = 0;

  if (dimension == 2) {
    int np1,np2;

    flag = (int) ubuf(buf[m++]).i;
    if (flag == 0) atom2connect[nlocal] = -1;
    else {
      if (nlocal_connect == nmax_connect) grow_connect();

      np1 = (int) ubuf(buf[m++]).i;
      np2 = (int) ubuf(buf[m++]).i;
      connect2d[nlocal_connect].np1 = np1;
      connect2d[nlocal_connect].np2 = np2;

      if (np1) {
        connect2d[nlocal_connect].neigh_p1 = tcp->get(np1,pool2d[nlocal_connect].neigh_p1);
        connect2d[nlocal_connect].pwhich_p1 = tcp->get(np1,pool2d[nlocal_connect].pwhich_p1);
        connect2d[nlocal_connect].nside_p1 = tcp->get(np1,pool2d[nlocal_connect].nside_p1);
        connect2d[nlocal_connect].aflag_p1 = tcp->get(np1,pool2d[nlocal_connect].aflag_p1);
        for (k = 0; k < np1; k++) {
          connect2d[nlocal_connect].neigh_p1[k] = (tagint) ubuf(buf[m++]).i;
          connect2d[nlocal_connect].pwhich_p1[k] = (tagint) ubuf(buf[m++]).i;
          connect2d[nlocal_connect].nside_p1[k] = (tagint) ubuf(buf[m++]).i;
          connect2d[nlocal_connect].aflag_p1[k] = (tagint) ubuf(buf[m++]).i;
        }
      } else {
        connect2d[nlocal_connect].neigh_p1 = nullptr;
        connect2d[nlocal_connect].pwhich_p1 = nullptr;
        connect2d[nlocal_connect].nside_p1 = nullptr;
        connect2d[nlocal_connect].aflag_p1 = nullptr;
      }

      if (np2) {
        connect2d[nlocal_connect].neigh_p2 = tcp->get(np2,pool2d[nlocal_connect].neigh_p2);
        connect2d[nlocal_connect].pwhich_p2 = tcp->get(np2,pool2d[nlocal_connect].pwhich_p2);
        connect2d[nlocal_connect].nside_p2 = tcp->get(np2,pool2d[nlocal_connect].nside_p2);
        connect2d[nlocal_connect].aflag_p2 = tcp->get(np2,pool2d[nlocal_connect].aflag_p2);
        for (k = 0; k < np2; k++) {
          connect2d[nlocal_connect].neigh_p2[k] = (tagint) ubuf(buf[m++]).i;
          connect2d[nlocal_connect].pwhich_p2[k] = (tagint) ubuf(buf[m++]).i;
          connect2d[nlocal_connect].nside_p2[k] = (tagint) ubuf(buf[m++]).i;
          connect2d[nlocal_connect].aflag_p2[k] = (tagint) ubuf(buf[m++]).i;
        }
      } else {
        connect2d[nlocal_connect].neigh_p2 = nullptr;
        connect2d[nlocal_connect].pwhich_p2 = nullptr;
        connect2d[nlocal_connect].nside_p2 = nullptr;
        connect2d[nlocal_connect].aflag_p2 = nullptr;
      }

      connect2atom[nlocal_connect] = nlocal;
      atom2connect[nlocal] = nlocal_connect++;
    }

  } else {
    int ne1,ne2,ne3,nc1,nc2,nc3;

    flag = (int) ubuf(buf[m++]).i;
    if (flag == 0) atom2connect[nlocal] = -1;
    else {
      if (nlocal_connect == nmax_connect) grow_connect();

      ne1 = (int) ubuf(buf[m++]).i;
      ne2 = (int) ubuf(buf[m++]).i;
      ne3 = (int) ubuf(buf[m++]).i;
      connect3d[nlocal_connect].ne1 = ne1;
      connect3d[nlocal_connect].ne2 = ne2;
      connect3d[nlocal_connect].ne3 = ne3;

      if (ne1) {
        connect3d[nlocal_connect].neigh_e1 = tcp->get(ne1,pool3d[nlocal_connect].neigh_e1);
        connect3d[nlocal_connect].ewhich_e1 = tcp->get(ne1,pool3d[nlocal_connect].ewhich_e1);
        connect3d[nlocal_connect].nside_e1 = tcp->get(ne1,pool3d[nlocal_connect].nside_e1);
        connect3d[nlocal_connect].aflag_e1 = tcp->get(ne1,pool3d[nlocal_connect].aflag_e1);
        for (k = 0; k < ne1; k++) {
          connect3d[nlocal_connect].neigh_e1[k] = (tagint) ubuf(buf[m++]).i;
          connect3d[nlocal_connect].ewhich_e1[k] = (tagint) ubuf(buf[m++]).i;
          connect3d[nlocal_connect].nside_e1[k] = (tagint) ubuf(buf[m++]).i;
          connect3d[nlocal_connect].aflag_e1[k] = (tagint) ubuf(buf[m++]).i;
        }
      } else {
        connect3d[nlocal_connect].neigh_e1 = nullptr;
        connect3d[nlocal_connect].ewhich_e1 = nullptr;
        connect3d[nlocal_connect].nside_e1 = nullptr;
        connect3d[nlocal_connect].aflag_e1 = nullptr;
      }

      if (ne2) {
        connect3d[nlocal_connect].neigh_e2 = tcp->get(ne2,pool3d[nlocal_connect].neigh_e2);
        connect3d[nlocal_connect].ewhich_e2 = tcp->get(ne2,pool3d[nlocal_connect].ewhich_e2);
        connect3d[nlocal_connect].nside_e2 = tcp->get(ne2,pool3d[nlocal_connect].nside_e2);
        connect3d[nlocal_connect].aflag_e2 = tcp->get(ne2,pool3d[nlocal_connect].aflag_e2);
        for (k = 0; k < ne2; k++) {
          connect3d[nlocal_connect].neigh_e2[k] = (tagint) ubuf(buf[m++]).i;
          connect3d[nlocal_connect].ewhich_e2[k] = (tagint) ubuf(buf[m++]).i;
          connect3d[nlocal_connect].nside_e2[k] = (tagint) ubuf(buf[m++]).i;
          connect3d[nlocal_connect].aflag_e2[k] = (tagint) ubuf(buf[m++]).i;
        }
      } else {
        connect3d[nlocal_connect].neigh_e2 = nullptr;
        connect3d[nlocal_connect].ewhich_e2 = nullptr;
        connect3d[nlocal_connect].nside_e2 = nullptr;
        connect3d[nlocal_connect].aflag_e2 = nullptr;
      }

      if (ne3) {
        connect3d[nlocal_connect].neigh_e3 = tcp->get(ne3,pool3d[nlocal_connect].neigh_e3);
        connect3d[nlocal_connect].ewhich_e3 = tcp->get(ne3,pool3d[nlocal_connect].ewhich_e3);
        connect3d[nlocal_connect].nside_e3 = tcp->get(ne3,pool3d[nlocal_connect].nside_e3);
        connect3d[nlocal_connect].aflag_e3 = tcp->get(ne3,pool3d[nlocal_connect].aflag_e3);
        for (k = 0; k < ne3; k++) {
          connect3d[nlocal_connect].neigh_e3[k] = (tagint) ubuf(buf[m++]).i;
          connect3d[nlocal_connect].ewhich_e3[k] = (tagint) ubuf(buf[m++]).i;
          connect3d[nlocal_connect].nside_e3[k] = (tagint) ubuf(buf[m++]).i;
          connect3d[nlocal_connect].aflag_e3[k] = (tagint) ubuf(buf[m++]).i;
        }
      } else {
        connect3d[nlocal_connect].neigh_e3 = nullptr;
        connect3d[nlocal_connect].ewhich_e3 = nullptr;
        connect3d[nlocal_connect].nside_e3 = nullptr;
        connect3d[nlocal_connect].aflag_e3 = nullptr;
      }

      nc1 = (int) ubuf(buf[m++]).i;
      nc2 = (int) ubuf(buf[m++]).i;
      nc3 = (int) ubuf(buf[m++]).i;
      connect3d[nlocal_connect].nc1 = nc1;
      connect3d[nlocal_connect].nc2 = nc2;
      connect3d[nlocal_connect].nc3 = nc3;
      
      if (nc1) {
        connect3d[nlocal_connect].neigh_c1 = tcp->get(nc1,pool3d[nlocal_connect].neigh_c1);
        connect3d[nlocal_connect].cwhich_c1 = tcp->get(nc1,pool3d[nlocal_connect].cwhich_c1);
        for (k = 0; k < nc1; k++) {
          connect3d[nlocal_connect].neigh_c1[k] = (tagint) ubuf(buf[m++]).i;
          connect3d[nlocal_connect].cwhich_c1[k] = (tagint) ubuf(buf[m++]).i;
        }
      } else {
        connect3d[nlocal_connect].neigh_c1 = nullptr;
        connect3d[nlocal_connect].cwhich_c1 = nullptr;
      }
      
      if (nc2) {
        connect3d[nlocal_connect].neigh_c2 = tcp->get(nc2,pool3d[nlocal_connect].neigh_c2);
        connect3d[nlocal_connect].cwhich_c2 = tcp->get(nc2,pool3d[nlocal_connect].cwhich_c2);
        for (k = 0; k < nc2; k++) {
          connect3d[nlocal_connect].neigh_c2[k] = (tagint) ubuf(buf[m++]).i;
          connect3d[nlocal_connect].cwhich_c2[k] = (tagint) ubuf(buf[m++]).i;
        }
      } else {
        connect3d[nlocal_connect].neigh_c2 = nullptr;
        connect3d[nlocal_connect].cwhich_c2 = nullptr;
      }
      
      if (nc3) {
        connect3d[nlocal_connect].neigh_c3 = tcp->get(nc3,pool3d[nlocal_connect].neigh_c3);
        connect3d[nlocal_connect].cwhich_c3 = tcp->get(nc3,pool3d[nlocal_connect].cwhich_c3);
        for (k = 0; k < nc3; k++) {
          connect3d[nlocal_connect].neigh_c3[k] = (tagint) ubuf(buf[m++]).i;
          connect3d[nlocal_connect].cwhich_c3[k] = (tagint) ubuf(buf[m++]).i;
        }
      } else {
        connect3d[nlocal_connect].neigh_c3 = nullptr;
        connect3d[nlocal_connect].cwhich_c3 = nullptr;
      }
      
      connect2atom[nlocal_connect] = nlocal;
      atom2connect[nlocal] = nlocal_connect++;
    }
  }

  return m;
}

/* ----------------------------------------------------------------------
   one-time forward comm of connectivity info from owned to ghost particles
   same pack/unpack operations as comm->borders() uses
------------------------------------------------------------------------- */

int FixSurfaceLocal::pack_forward_comm(int n, int *list, double *buf,
                                       int /*pbc_flag*/, int * /*pbc*/)
{
  int m = pack_border(n,list,buf);
  return m;
}

/* ---------------------------------------------------------------------- */

void FixSurfaceLocal::unpack_forward_comm(int n, int first, double *buf)
{
  int m = unpack_border(n,first,buf);
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixSurfaceLocal::memory_usage()
{
  double bytes = 0.0;
  
  if (dimension == 2) {
    bytes = nmax_connect*sizeof(Connect2d);
    bytes = nmax_connect*sizeof(Pool2d);
  } else {
    bytes = nmax_connect*sizeof(Connect3d);
    bytes = nmax_connect*sizeof(Pool3d);
  }
  
  bytes += atom->nmax * sizeof(int);               // atom2connect vector
  bytes += nmax_connect * sizeof(int);             // connect2atom vector

  return bytes;
}

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// methods for distributed connectivity builds in 2d or 3d
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------

/* ----------------------------------------------------------------------
   create and initialize Connect2d info for owned lines
     only np1,np2 and neigh_p1,neigh_p2
     also atom2connect,connect2atom
   this must be done with INEXACT point matching
     done via Rvous algorithm on set of slightly overlapping bins
   inexact b/c once datafile is read, lines have only a center point, length, and theta
     thus same point calculated by 2 lines may be epsilon different
------------------------------------------------------------------------- */

void FixSurfaceLocal::connectivity2d_local()
{
  int i,j,k,m,n;

  avec_line = (AtomVecLine *) atom->style_match("line");
  if (!avec_line)
    error->all(FLERR,"Fix surface/local NULL requires atom style line");

  // calculate epssq = square of EPSILON fraction of minimum line length
  // for use in point_match()
  
  if (epssq < 0.0) epsilon_calculate();

  // count owned lines
  // error check for no lines on any proc

  int *line = atom->line;
  int nlocal = atom->nlocal;
  
  int nline = 0;
  for (i = 0; i < nlocal; i++)
    if (line[i] >= 0) nline++;

  int anyline;
  MPI_Allreduce(&nline,&anyline,1,MPI_INT,MPI_MAX,world);

  if (!anyline) error->all(FLERR,"Fix surface/local NULL requires line particles exist");

  // allocate connection info for owned lines
  // initialize atom2connect for both particles and lines

  nlocal_connect = nmax_connect = nline;
  nghost_connect = 0;
  grow_connect();

  for (i = 0; i < nlocal; i++) {
    atom2connect[i] = line[i];
    if (line[i] < 0) continue;
    j = line[i];
    connect2atom[j] = i;
  }

  // calculate current endpts of owned lines

  double **endpts;
  memory->create(endpts,nline,4,"surface/local:endpts");
  calculate_endpts(endpts);

  // compute min/max extent of my line endpts
  // MPI_Allreduce for bbox of all lines

  double mylo[2],myhi[2];

  mylo[0] = mylo[1] = BIG;
  myhi[0] = myhi[1] = -BIG;

  m = 0;
  for (i = 0; i < nlocal; i++) {
    if (line[i] < 0) continue;
    mylo[0] = MIN(mylo[0],endpts[m][0]);
    myhi[0] = MAX(myhi[0],endpts[m][0]);
    mylo[1] = MIN(mylo[1],endpts[m][1]);
    myhi[1] = MAX(myhi[1],endpts[m][1]);
    mylo[0] = MIN(mylo[0],endpts[m][2]);
    myhi[0] = MAX(myhi[0],endpts[m][2]);
    mylo[1] = MIN(mylo[1],endpts[m][3]);
    myhi[1] = MAX(myhi[1],endpts[m][3]);
    m++;
  }

  MPI_Allreduce(mylo,bboxlo,2,MPI_DOUBLE,MPI_MIN,world);
  MPI_Allreduce(myhi,bboxhi,2,MPI_DOUBLE,MPI_MAX,world);

  // add 2*EPS to all 4 edges of bbox

  bboxlo[0] -= 2.0*eps;
  bboxlo[1] -= 2.0*eps;
  bboxhi[0] += 2.0*eps;
  bboxhi[1] += 2.0*eps;

  // conceptual binning of bbox by up to NBIN x NBIN bins
  // ensure bin size is not <= 2*EPS so that pt +/- EPS cannot overlap > 4 bins
  // nbin xy = # of bins in each dim

  nbinx = static_cast<int> ((bboxhi[0]-bboxlo[0])/(4.0*eps));
  nbinx = MIN(nbinx,NBIN);
  nbinx = MAX(nbinx,1);
  nbiny = static_cast<int> ((bboxhi[1]-bboxlo[1])/(4.0*eps));
  nbiny = MIN(nbiny,NBIN);
  nbiny = MAX(nbiny,1);

  nbins = nbinx * nbiny;

  invbinx = nbinx / (bboxhi[0] - bboxlo[0]);
  invbiny = nbiny / (bboxhi[1] - bboxlo[1]);

  // inbuf = list of datums to send to procs in Rvous decomposition of bins
  // every Pth bin is assigned to each proc
  // use overlap_bins_2d() to find all bins a line endpt overlaps within EPS
  // allows for matching pts in Rvous decomp which are up to EPS apart
  
  int me = comm->me;
  int nprocs = comm->nprocs;

  tagint *tag = atom->tag;

  int *proclist = nullptr;
  InRvous *inbuf = nullptr;
  int ncount = 0;
  int maxcount = 0;

  int indices[4];

  m = 0;
  for (i = 0; i < nlocal; i++) {
    if (line[i] < 0) continue;

    for (int ipoint = 0; ipoint < 2; ipoint++) {
      n = overlap_bins_2d(&endpts[m][2*ipoint],eps,indices);

      if (ncount+n > maxcount) {
        maxcount += DELTA_RVOUS;
        memory->grow(proclist,maxcount,"fix/surface/local:proclist");
        inbuf = (InRvous *) memory->srealloc(inbuf,maxcount*sizeof(InRvous),"surface/local:inbuf");
      }

      for (k = 0; k < n; k++) {
        proclist[ncount] = indices[k] % nprocs;
        inbuf[ncount].proc = me;
        inbuf[ncount].ibin = indices[k];
        inbuf[ncount].ilocal = i;
        inbuf[ncount].ipoint = ipoint;
        inbuf[ncount].atomID = tag[i];
        inbuf[ncount].x[0] = endpts[m][2*ipoint];
        inbuf[ncount].x[1] = endpts[m][2*ipoint+1];
        inbuf[ncount].x[2] = 0.0;
        ncount++;
      }
    }

    m++;
  }

  memory->destroy(endpts);

  // perform rendezvous operation
  // each proc owns every Pth bin
  // receives all points in those bins

  char *buf;
  int nreturn = comm->rendezvous(RVOUS,ncount,(char *) inbuf,sizeof(InRvous),
                                 0,proclist,
                                 point_match,0,buf,sizeof(OutRvous),
                                 (void *) this);
  auto outbuf = (OutRvous *) buf;

  memory->destroy(proclist);
  memory->sfree(inbuf);

  // loop over received Rvous datums
  // p12_counts = # of connections for each point on my lines
  // this will overcount (potentially by 4x) due to bins overlapping by EPS
  // datums do NOT include self connection

  int *p1_counts,*p2_counts;
  memory->create(p1_counts,nlocal_connect,"surface/local:p1_counts");
  memory->create(p2_counts,nlocal_connect,"surface/local:p2_counts");

  int ilocal,iline;

  for (i = 0; i < nlocal_connect; i++)
    p1_counts[i] = p2_counts[i] = 0;

  for (i = 0; i < nreturn; i++) {
    ilocal = outbuf[i].ilocal;
    iline = line[ilocal];
    if (outbuf[i].ipoint == 0) p1_counts[iline]++;
    else p2_counts[iline]++;
  }

  // allocate ragged neigh_p12 vectors using p12_counts
  // this will overallocate because of bins overlapping by EPS
  //   will reallocate below when know exact size

  tagint **neigh_p1,**neigh_p2;
  memory->create_ragged(neigh_p1,nlocal_connect,p1_counts,"surface/local:neigh_p1");
  memory->create_ragged(neigh_p2,nlocal_connect,p2_counts,"surface/local:neigh_p2");

  // loop over received Rvous datums
  // add each atomID to neigh_p12 but only if not already in the list
  // recalculate exact p12_counts

  for (i = 0; i < nlocal_connect; i++)
    p1_counts[i] = p2_counts[i] = 0;

  int np;
  tagint atomID;
  tagint *neigh;
  
  for (i = 0; i < nreturn; i++) {
    ilocal = outbuf[i].ilocal;
    iline = line[ilocal];
    if (outbuf[i].ipoint == 0) {
      atomID = outbuf[i].atomID;
      np = p1_counts[iline];
      neigh = neigh_p1[iline];
      for (j = 0; j < np; j++)
        if (neigh[j] == atomID) break;
      if (j == np) {
        neigh[np] = atomID;
        p1_counts[iline]++;
      }
    } else {
      atomID = outbuf[i].atomID;
      np = p2_counts[iline];
      neigh = neigh_p1[iline];
      for (j = 1; j < np; j++)
        if (neigh[j] == atomID) break;
      if (j == np) {
        neigh[np] = atomID;
        p2_counts[iline];
      }
    }
  }
  
  // set np1,np2 via exact p12_counts
  // likewise allocate all vectors within Connect2d
  // use ragged neigh_p12 to set neigh_p12 within Connect2d
  // other vectors will be set in connectivity2d_complete()

  for (i = 0; i < nlocal_connect; i++) {
    connect2d[i].np1 = p1_counts[i];;
    if (connect2d[i].np1) {
      connect2d[i].neigh_p1 = tcp->get(connect2d[i].np1,pool2d[i].neigh_p1);
      connect2d[i].pwhich_p1 = tcp->get(connect2d[i].np1,pool2d[i].pwhich_p1);
      connect2d[i].nside_p1 = tcp->get(connect2d[i].np1,pool2d[i].nside_p1);
      connect2d[i].aflag_p1 = tcp->get(connect2d[i].np1,pool2d[i].aflag_p1);
      for (j = 0; j < connect2d[i].np1; j++)
        connect2d[i].neigh_p1[j] = neigh_p1[i][j];
    } else {
      connect2d[i].neigh_p1 = nullptr;
      connect2d[i].pwhich_p1 = nullptr;
      connect2d[i].nside_p1 = nullptr;
      connect2d[i].aflag_p1 = nullptr;
    }
    
    connect2d[i].np2 = p2_counts[i];;
    if (connect2d[i].np2) {
      connect2d[i].neigh_p2 = tcp->get(connect2d[i].np2,pool2d[i].neigh_p2);
      connect2d[i].pwhich_p2 = tcp->get(connect2d[i].np2,pool2d[i].pwhich_p2);
      connect2d[i].nside_p2 = tcp->get(connect2d[i].np2,pool2d[i].nside_p2);
      connect2d[i].aflag_p2 = tcp->get(connect2d[i].np2,pool2d[i].aflag_p2);
      for (j = 0; j < connect2d[i].np2; j++)
        connect2d[i].neigh_p2[j] = neigh_p2[i][j];
    } else {
      connect2d[i].neigh_p2 = nullptr;
      connect2d[i].pwhich_p2 = nullptr;
      connect2d[i].nside_p2 = nullptr;
      connect2d[i].aflag_p2 = nullptr;
    }
  }

  // clean up
  
  memory->sfree(outbuf);
  memory->destroy(p1_counts);
  memory->destroy(p2_counts);
  memory->destroy(neigh_p1);
  memory->destroy(neigh_p2);
}

/* ----------------------------------------------------------------------
   create and initialize Connect3d info for owned tris
     only ne1,ne2,ne3 and neigh_e1,neigh_e2,neigh_e3
     only nc1,nc2,nc3 and neigh_c1,neigh_c2,neigh_c3
     also atom2connect,connect2atom
   this must be done with INEXACT point matching
     done via Rvous algorithm on set of slightly overlapping bins
   inexact b/c once datafile is read, tris have a center point, quaternion,
     and body-frame corner point displacements
     thus same point calculated by 2 tris may be epsilon different
------------------------------------------------------------------------- */

void FixSurfaceLocal::connectivity3d_local()
{
  int i,j,k,m,n;

  avec_tri = (AtomVecTri *) atom->style_match("tri");
  if (!avec_tri)
    error->all(FLERR,"Fix surface/local NULL requires atom style tri");

  // epssq = square of EPSILON fraction of minimum tri diameter
  // for use in point_match()
  
  if (epssq < 0.0) epsilon_calculate();
  
  // count owned triangles
  // error check for no tris on any proc

  int *tri = atom->tri;
  int nlocal = atom->nlocal;

  int ntri = 0;
  for (i = 0; i < nlocal; i++)
    if (tri[i] >= 0) ntri++;

  int anytri;
  MPI_Allreduce(&ntri,&anytri,1,MPI_INT,MPI_MAX,world);

  if (!anytri) error->all(FLERR,"Fix surface/local NULL requires tri particles exist");

  // allocate connection info for owned triangles
  // initialize atom2connect for both particles and triangles

  nlocal_connect = nmax_connect = ntri;
  nghost_connect = 0;
  grow_connect();

  for (i = 0; i < nlocal; i++) {
    atom2connect[i] = tri[i];
    if (tri[i] < 0) continue;
    j = tri[i];
    connect2atom[j] = i;
  }

  // calculate current corners of owned tris

  double **corners;
  memory->create(corners,ntri,9,"surface/local:corners");
  calculate_corners(corners);

  // compute min/max extent of my tri corners
  // MPI_Allreduce for bbox of all lines

  double mylo[3],myhi[3];

  mylo[0] = mylo[1] = mylo[2] = BIG;
  myhi[0] = myhi[1] = myhi[2] = -BIG;

  m = 0;
  for (i = 0; i < nlocal; i++) {
    if (tri[i] < 0) continue;
    k = 0;
    for (j = 0; j < 3; j++) {
      mylo[0] = MIN(mylo[0],corners[m][k]);
      myhi[0] = MAX(myhi[0],corners[m][k]);
      mylo[1] = MIN(mylo[1],corners[m][k+1]);
      myhi[1] = MAX(myhi[1],corners[m][k+1]);
      mylo[2] = MIN(mylo[2],corners[m][k+2]);
      myhi[2] = MAX(myhi[2],corners[m][k+2]);
      k += 3;
    }
    m++;
  }

  MPI_Allreduce(mylo,bboxlo,3,MPI_DOUBLE,MPI_MIN,world);
  MPI_Allreduce(myhi,bboxhi,3,MPI_DOUBLE,MPI_MAX,world);

  // add 2*EPS to all 4 faces of bbox

  bboxlo[0] -= 2.0*eps;
  bboxlo[1] -= 2.0*eps;
  bboxlo[2] -= 2.0*eps;
  bboxhi[0] += 2.0*eps;
  bboxhi[1] += 2.0*eps;
  bboxhi[2] += 2.0*eps;

  // conceptual binning of bbox by up to NBIN x NBIN x NBIN bins
  // ensure bin size is not <= 2*EPS so that pt +/- EPS cannot overlap > 8 bins
  // nbin xyz = # of bins in each dim

  nbinx = static_cast<int> ((bboxhi[0]-bboxlo[0])/(4.0*eps));
  nbinx = MIN(nbinx,NBIN);
  nbinx = MAX(nbinx,1);
  nbiny = static_cast<int> ((bboxhi[1]-bboxlo[1])/(4.0*eps));
  nbiny = MIN(nbiny,NBIN);
  nbiny = MAX(nbiny,1);
  nbinz = static_cast<int> ((bboxhi[2]-bboxlo[2])/(4.0*eps));
  nbinz = MIN(nbinz,NBIN);
  nbinz = MAX(nbinz,1);

  nbins = nbinx * nbiny * nbinz;

  invbinx = nbinx / (bboxhi[0] - bboxlo[0]);
  invbiny = nbiny / (bboxhi[1] - bboxlo[1]);
  invbinz = nbinz / (bboxhi[2] - bboxlo[2]);

  // inbuf = list of datums to send to procs in Rvous decomposition of bins
  // every Pth bin is assigned to each proc
  // use overlap_bins_3d() to find all bins a tri corner pt overlaps within EPS
  // allows for matching pts in Rvous decomp which are up to EPS apart

  int me = comm->me;
  int nprocs = comm->nprocs;

  tagint *tag = atom->tag;

  int *proclist = nullptr;
  InRvous *inbuf = nullptr;
  int ncount = 0;
  int maxcount = 0;

  int indices[8];

  m = 0;
  for (i = 0; i < nlocal; i++) {
    if (tri[i] < 0) continue;

    for (int ipoint = 0; ipoint < 3; ipoint++) {
      n = overlap_bins_3d(&corners[m][3*ipoint],eps,indices);

      if (ncount+n > maxcount) {
        maxcount += DELTA_RVOUS;
        memory->grow(proclist,maxcount,"fix/surface/local:proclist");
        inbuf = (InRvous *) memory->srealloc(inbuf,maxcount*sizeof(InRvous),"surface/local:inbuf");
      }

      for (k = 0; k < n; k++) {
        proclist[ncount] = indices[k] % nprocs;
        inbuf[ncount].proc = me;
        inbuf[ncount].ibin = indices[k];
        inbuf[ncount].ilocal = i;
        inbuf[ncount].ipoint = ipoint;
        inbuf[ncount].atomID = tag[i];
        inbuf[ncount].x[0] = corners[m][3*ipoint];
        inbuf[ncount].x[1] = corners[m][3*ipoint+1];
        inbuf[ncount].x[2] = corners[m][3*ipoint+2];
        ncount++;
      }
    }

    m++;
  }

  memory->destroy(corners);

  // perform rendezvous operation
  // each proc owns every Pth bin
  // receives all points in those bins

  char *buf;
  int nreturn = comm->rendezvous(RVOUS,ncount,(char *) inbuf,sizeof(InRvous),
                                 0,proclist,
                                 point_match,0,buf,sizeof(OutRvous),
                                 (void *) this);
  auto outbuf = (OutRvous *) buf;

  memory->destroy(proclist);
  memory->sfree(inbuf);

  // loop over received Rvous datums
  // n1/n2/n3_counts = # of connections for each corner point on my tris
  // this will overcount (potentially by 8x) due to bins overlapping by EPS
  // datums do NOT include self connection

  int *n1_counts,*n2_counts,*n3_counts;
  memory->create(n1_counts,nlocal_connect,"surface/local:n1_counts");
  memory->create(n2_counts,nlocal_connect,"surface/local:n2_counts");
  memory->create(n3_counts,nlocal_connect,"surface/local:n3_counts");

  int ilocal,iconnect;

  for (i = 0; i < nlocal_connect; i++)
    n1_counts[i] = n2_counts[i] = n3_counts[i] = 0;

  for (i = 0; i < nreturn; i++) {
    ilocal = outbuf[i].ilocal;
    iconnect = atom2connect[ilocal];
    if (outbuf[i].ipoint == 0) n1_counts[iconnect]++;
    else if (outbuf[i].ipoint == 1) n2_counts[iconnect]++;
    else n3_counts[iconnect]++;
  }
  
  // allocate ragged neigh_n123 vectors using n1/n2/n3_counts
  // this will overallocate because of bins overlapping by EPS
  //   will reallocate below when know exact size of edge and corner neighs

  tagint **neigh_n1,**neigh_n2,**neigh_n3;
  memory->create_ragged(neigh_n1,nlocal_connect,n1_counts,"surface/local:neigh_n1");
  memory->create_ragged(neigh_n2,nlocal_connect,n2_counts,"surface/local:neigh_n2");
  memory->create_ragged(neigh_n3,nlocal_connect,n3_counts,"surface/local:neigh_n3");

  // loop over received Rvous datums
  // add each atomID to neigh_n1/n2/n3 but only if not already in the list
  // recalculate exact n1/n2/n3_counts

  for (i = 0; i < nlocal_connect; i++)
    n1_counts[i] = n2_counts[i] = n3_counts[i] = 0;

  int np;
  tagint atomID;
  tagint *neigh;
  
  for (i = 0; i < nreturn; i++) {
    ilocal = outbuf[i].ilocal;
    iconnect = atom2connect[ilocal];
    if (outbuf[i].ipoint == 0) {
      atomID = outbuf[i].atomID;
      np = n1_counts[iconnect];
      neigh = neigh_n1[iconnect];
      for (j = 0; j < np; j++)
        if (neigh[j] == atomID) break;
      if (j == np) {
        neigh[np] = atomID;
        n1_counts[iconnect]++;
      }
    } else if (outbuf[i].ipoint == 1) {
      atomID = outbuf[i].atomID;
      np = n2_counts[iconnect];
      neigh = neigh_n2[iconnect];
      for (j = 0; j < np; j++)
        if (neigh[j] == atomID) break;
      if (j == np) {
        neigh[np] = atomID;
        n2_counts[iconnect]++;
      }
    } else if (outbuf[i].ipoint == 2) {
      atomID = outbuf[i].atomID;
      np = n3_counts[iconnect];
      neigh = neigh_n3[iconnect];
      for (j = 0; j < np; j++)
        if (neigh[j] == atomID) break;
      if (j == np) {
        neigh[np] = atomID;
        n3_counts[iconnect]++;
      }
    }
  }

  memory->sfree(outbuf);

  // now have exact list of all neighbor tris of each corner point
  // each neighbor is either an edge neighbor or corner neighbor
  // edge neighbor = appears in neigh lists of 2 adjacent corners
  // corner neighbor = appears only in neigh list of a single corner
  
  int n1,n2;
  tagint *neigh1,*neigh2;

  for (i = 0; i < nlocal_connect; i++) {
    connect3d[i].ne1 = connect3d[i].ne2 = connect3d[i].ne3 = 0;
    connect3d[i].nc1 = connect3d[i].nc2 = connect3d[i].nc3 = 0;
  }
  
  for (i = 0; i < nlocal_connect; i++) {

    // count edge neighbors first

    n1 = n1_counts[i];
    n2 = n2_counts[i];
    neigh1 = neigh_n1[i];
    neigh2 = neigh_n2[i];
    for (j = 0; j < n1; j++) {
      for (k = 0; k < n2; k++) {
        if (neigh2[k] == neigh1[j]) {
          connect3d[i].ne1++;
          break;
        }
      }
    }

    n1 = n2_counts[i];
    n2 = n3_counts[i];
    neigh1 = neigh_n2[i];
    neigh2 = neigh_n3[i];
    for (j = 0; j < n1; j++) {
      for (k = 0; k < n2; k++) {
        if (neigh2[k] == neigh1[j]) {
          connect3d[i].ne2++;
          break;
        }
      }
    }

    n1 = n3_counts[i];
    n2 = n1_counts[i];
    neigh1 = neigh_n3[i];
    neigh2 = neigh_n1[i];
    for (j = 0; j < n1; j++) {
      for (k = 0; k < n2; k++) {
        if (neigh2[k] == neigh1[j]) {
          connect3d[i].ne3++;
          break;
        }
      }
    }

    // corner neighbor count = all neighbors minus 2 sets of edge neighbors

    connect3d[i].nc1 = n1_counts[i] - connect3d[i].ne1 - connect3d[i].ne3;
    connect3d[i].nc2 = n2_counts[i] - connect3d[i].ne2 - connect3d[i].ne1;
    connect3d[i].nc3 = n3_counts[i] - connect3d[i].ne3 - connect3d[i].ne2;
  }

  // use exact n1/n2/n3_counts to allocate all vectors within Connect3d

  for (i = 0; i < nlocal_connect; i++) {
    if (connect3d[i].ne1) {
      connect3d[i].neigh_e1 = tcp->get(connect3d[i].ne1,pool3d[i].neigh_e1);
      connect3d[i].ewhich_e1 = tcp->get(connect3d[i].ne1,pool3d[i].ewhich_e1);
      connect3d[i].nside_e1 = tcp->get(connect3d[i].ne1,pool3d[i].nside_e1);
      connect3d[i].aflag_e1 = tcp->get(connect3d[i].ne1,pool3d[i].aflag_e1);
    } else {
      connect3d[i].neigh_e1 = nullptr;
      connect3d[i].ewhich_e1 = nullptr;
      connect3d[i].nside_e1 = nullptr;
      connect3d[i].aflag_e1 = nullptr;
    }
    if (connect3d[i].ne2) {
      connect3d[i].neigh_e2 = tcp->get(connect3d[i].ne2,pool3d[i].neigh_e2);
      connect3d[i].ewhich_e2 = tcp->get(connect3d[i].ne2,pool3d[i].ewhich_e2);
      connect3d[i].nside_e2 = tcp->get(connect3d[i].ne2,pool3d[i].nside_e2);
      connect3d[i].aflag_e2 = tcp->get(connect3d[i].ne2,pool3d[i].aflag_e2);
    } else {
      connect3d[i].neigh_e2 = nullptr;
      connect3d[i].ewhich_e2 = nullptr;
      connect3d[i].nside_e2 = nullptr;
      connect3d[i].aflag_e2 = nullptr;
    }
    if (connect3d[i].ne3) {
      connect3d[i].neigh_e3 = tcp->get(connect3d[i].ne3,pool3d[i].neigh_e3);
      connect3d[i].ewhich_e3 = tcp->get(connect3d[i].ne3,pool3d[i].ewhich_e3);
      connect3d[i].nside_e3 = tcp->get(connect3d[i].ne3,pool3d[i].nside_e3);
      connect3d[i].aflag_e3 = tcp->get(connect3d[i].ne3,pool3d[i].aflag_e3);
    } else {
      connect3d[i].neigh_e3 = nullptr;
      connect3d[i].ewhich_e3 = nullptr;
      connect3d[i].nside_e3 = nullptr;
      connect3d[i].aflag_e3 = nullptr;
    }

    if (connect3d[i].nc1) {
      connect3d[i].neigh_c1 = tcp->get(connect3d[i].nc1,pool3d[i].neigh_c1);
      connect3d[i].cwhich_c1 = tcp->get(connect3d[i].nc1,pool3d[i].cwhich_c1);
    } else {
      connect3d[i].neigh_c1 = nullptr;
      connect3d[i].cwhich_c1 = nullptr;
    }
    if (connect3d[i].nc2) {
      connect3d[i].neigh_c2 = tcp->get(connect3d[i].nc2,pool3d[i].neigh_c2);
      connect3d[i].cwhich_c2 = tcp->get(connect3d[i].nc2,pool3d[i].cwhich_c2);
    } else {
      connect3d[i].neigh_c2 = nullptr;
      connect3d[i].cwhich_c2 = nullptr;
    }
    if (connect3d[i].nc3) {
      connect3d[i].neigh_c3 = tcp->get(connect3d[i].nc3,pool3d[i].neigh_c3);
      connect3d[i].cwhich_c3 = tcp->get(connect3d[i].nc3,pool3d[i].cwhich_c3);
    } else {
      connect3d[i].neigh_c3 = nullptr;
      connect3d[i].cwhich_c3 = nullptr;
    }
  }

  // populate neigh_e123 vectors within Connect3d

  for (i = 0; i < nlocal_connect; i++)
    connect3d[i].ne1 = connect3d[i].ne2 = connect3d[i].ne3 = 0;
  
  for (i = 0; i < nlocal_connect; i++) {
    n1 = n1_counts[i];
    n2 = n2_counts[i];
    neigh1 = neigh_n1[i];
    neigh2 = neigh_n2[i];
    for (j = 0; j < n1; j++) {
      for (k = 0; k < n2; k++) {
        if (neigh2[k] == neigh1[j]) {
          connect3d[i].neigh_e1[connect3d[i].ne1++] = neigh1[j];
          break;
        }
      }
    }

    n1 = n2_counts[i];
    n2 = n3_counts[i];
    neigh1 = neigh_n2[i];
    neigh2 = neigh_n3[i];
    for (j = 0; j < n1; j++) {
      for (k = 0; k < n2; k++) {
        if (neigh2[k] == neigh1[j]) {
          connect3d[i].neigh_e2[connect3d[i].ne2++] = neigh1[j];
          break;
        }
      }
    }

    n1 = n3_counts[i];
    n2 = n1_counts[i];
    neigh1 = neigh_n3[i];
    neigh2 = neigh_n1[i];
    for (j = 0; j < n1; j++) {
      for (k = 0; k < n2; k++) {
        if (neigh2[k] == neigh1[j]) {
          connect3d[i].neigh_e3[connect3d[i].ne3++] = neigh1[j];
          break;
        }
      }
    }
  }

  // populate neigh_c123 vectors within Connect3d

  for (i = 0; i < nlocal_connect; i++)
    connect3d[i].nc1 = connect3d[i].nc2 = connect3d[i].nc3 = 0;

  int flag;
  
  for (i = 0; i < nlocal_connect; i++) {
    n = n1_counts[i];
    neigh = neigh_n1[i];
    for (j = 0; j < n; j++) {
      flag = 0;
      for (k = 0; k < connect3d[i].ne1; k++) {
        if (neigh[j] == connect3d[i].neigh_e1[k]) {
          flag = 1;
          break;
        }
      }
      for (k = 0; k < connect3d[i].ne3; k++) {
        if (neigh[j] == connect3d[i].neigh_e3[k]) {
          flag = 1;
          break;
        }
      }
      if (!flag) connect3d[i].neigh_c1[connect3d[i].nc1++] = neigh[j];
    }

    n = n2_counts[i];
    neigh = neigh_n2[i];
    for (j = 0; j < n; j++) {
      flag = 0;
      for (k = 0; k < connect3d[i].ne2; k++) {
        if (neigh[j] == connect3d[i].neigh_e2[k]) {
          flag = 1;
          break;
        }
      }
      for (k = 0; k < connect3d[i].ne1; k++) {
        if (neigh[j] == connect3d[i].neigh_e1[k]) {
          flag = 1;
          break;
        }
      }
      if (!flag) connect3d[i].neigh_c2[connect3d[i].nc2++] = neigh[j];
    }

    n = n3_counts[i];
    neigh = neigh_n3[i];
    for (j = 0; j < n; j++) {
      flag = 0;
      for (k = 0; k < connect3d[i].ne3; k++) {
        if (neigh[j] == connect3d[i].neigh_e3[k]) {
          flag = 1;
          break;
        }
      }
      for (k = 0; k < connect3d[i].ne2; k++) {
        if (neigh[j] == connect3d[i].neigh_e2[k]) {
          flag = 1;
          break;
        }
      }
      if (!flag) connect3d[i].neigh_c3[connect3d[i].nc3++] = neigh[j];
    }
  }

  // clean up
  
  memory->destroy(n1_counts);
  memory->destroy(n2_counts);
  memory->destroy(n3_counts);
  memory->destroy(neigh_n1);
  memory->destroy(neigh_n2);
  memory->destroy(neigh_n3);
}

/* ----------------------------------------------------------------------
   callback from comm->rvous() for 2d or 3d
   operates in Rvous decomposition of bins
------------------------------------------------------------------------- */

int FixSurfaceLocal::point_match(int n, char *inbuf,
                                 int &rflag, int *&proclist, char *&outbuf,
                                 void *ptr)
{
  // access class data for epssq and bin count

  auto fslptr = (FixSurfaceLocal *) ptr;
  Memory *memory = fslptr->memory;
  Comm *comm = fslptr->comm;

  double epssq = fslptr->epssq;
  int nbins = fslptr->nbins;
  int nprocs = comm->nprocs;
  int me = comm->me;

  // nmine = # of bins I own
  // num[nmine] = count of datums in each bin
  // first[nmine] = index of 1st datum in each bin
  // next[n] = index to next datum in each bin

  int nmine = nbins / nprocs;
  if (me < nbins % nprocs) nmine++;

  int *num,*first,*next;
  memory->create(num,nmine,"surface/local:num");
  memory->create(first,nmine,"surface/local:first");
  memory->create(next,n,"surface/local:next");

  for (int i = 0; i < nmine; i++) num[i] = 0;
  for (int i = 0; i < nmine; i++) first[i] = -1;

  auto in = (InRvous *) inbuf;

  int i,j,ibin,whichbin;

  for (int i = n-1; i >= 0; i--) {
    ibin = in[i].ibin;
    whichbin = ibin / nprocs;
    if (first[whichbin] < 0) next[i] = -1;
    else next[i] = first[whichbin];
    first[whichbin] = i;
    num[whichbin]++;
  }

  // double loop over datums in each bin to to identify point matches
  // match = 2 points within EPS distance of each other
  // add each match to outbuf, to send atomID of other particle which matches

  proclist = nullptr;
  OutRvous *out = nullptr;
  int ncount = 0;
  int maxcount = 0;

  double dx,dy,dz,rsq;

  for (int ibin = 0; ibin < nmine; ibin++) {
    i = first[ibin];

    while (i >= 0) {
      j = first[ibin];
      while (j >= 0) {
        if (j == i) break;
        dx = in[i].x[0] - in[j].x[0];
        dy = in[i].x[1] - in[j].x[1];
        dz = in[i].x[2] - in[j].x[2];
        rsq = dx*dx + dy*dy + dz*dz;
        if (rsq < epssq) {
          if (ncount == maxcount) {
            maxcount += DELTA_RVOUS;
            memory->grow(proclist,maxcount,"surface/local:proclist");
            out = (OutRvous *)
              memory->srealloc(out,maxcount*sizeof(OutRvous),"surface/local:outbuf");
          }
          proclist[ncount] = in[i].proc;
          out[ncount].ilocal = in[i].ilocal;
          out[ncount].ipoint = in[i].ipoint;
          out[ncount].atomID = in[j].atomID;
          ncount++;
        }
        j = next[j];
      }
      i = next[i];
    }
  }

  // clean up

  memory->destroy(num);
  memory->destroy(first);
  memory->destroy(next);

  // return values

  rflag = 2;
  outbuf = (char *) out;
  return ncount;
}

/* ----------------------------------------------------------------------
   compute current end points of my owned lines
------------------------------------------------------------------------- */

void FixSurfaceLocal::calculate_endpts(double **endpts)
{
  double length,theta,dx,dy;

  AtomVecLine::Bonus *bonus = avec_line->bonus;
  double **x = atom->x;
  int *line = atom->line;
  int nlocal = atom->nlocal;

  int m = 0;
  for (int i = 0; i < nlocal; i++) {
    if (line[i] < 0) continue;
    length = bonus[line[i]].length;
    theta = bonus[line[i]].theta;
    dx = 0.5*length*cos(theta);
    dy = 0.5*length*sin(theta);
    endpts[m][0] = x[i][0] - dx;
    endpts[m][1] = x[i][1] - dy;
    endpts[m][2] = x[i][0] + dx;
    endpts[m][3] = x[i][1] + dy;
    m++;
  }
}

/* ----------------------------------------------------------------------
   calculate all bins which pt +/- EPS overlaps with
   return N = # of overlapped bins
   return indices = list of bin IDs (0 to Nbins-1)
------------------------------------------------------------------------- */

int FixSurfaceLocal::overlap_bins_2d(double *pt, double eps, int *indices)
{
  int ilo = static_cast<int> ((pt[0]-eps-bboxlo[0]) * invbinx);
  int ihi = static_cast<int> ((pt[0]+eps-bboxlo[0]) * invbinx);
  int jlo = static_cast<int> ((pt[1]-eps-bboxlo[1]) * invbiny);
  int jhi = static_cast<int> ((pt[1]+eps-bboxlo[1]) * invbiny);

  ilo = MAX(ilo,0);
  ihi = MIN(ihi,nbinx-1);
  jlo = MAX(jlo,0);
  jhi = MIN(jhi,nbiny-1);

  if (ilo == ihi && jlo == jhi) {
    indices[0] = jlo*nbinx + ilo;
    return 1;
  }

  int i,j;
  int n = 0;
  for (j = jlo; j <= jhi; j++)
    for (i = ilo; i <= ihi; i++)
      indices[n++] = j*nbinx + i;
  return n;
}

/* ----------------------------------------------------------------------
   compute current corner points of my owned triangles
------------------------------------------------------------------------- */

void FixSurfaceLocal::calculate_corners(double **corners)
{
  int ibonus;
  double p[3][3];
  double *corner;

  AtomVecTri::Bonus *bonus = avec_tri->bonus;
  double **x = atom->x;
  int *tri = atom->tri;
  int nlocal = atom->nlocal;

  int m = 0;
  for (int i = 0; i < nlocal; i++) {
    if (tri[i] < 0) continue;
    ibonus = tri[i];
    corner = corners[m];
    MathExtra::quat_to_mat(bonus[ibonus].quat,p);
    MathExtra::matvec(p,bonus[ibonus].c1,&corner[0]);
    MathExtra::add3(x[i],&corner[0],&corner[0]);
    MathExtra::matvec(p,bonus[ibonus].c2,&corner[3]);
    MathExtra::add3(x[i],&corner[3],&corner[3]);
    MathExtra::matvec(p,bonus[ibonus].c3,&corner[6]);
    MathExtra::add3(x[i],&corner[6],&corner[6]);
    m++;
  }
}

/* ----------------------------------------------------------------------
   map point +/ EPS in both dims to all bins it overlaps with
   return N = # of overlapped bins
   return indices = list of bin IDs, each from 0 to Nbins-1
------------------------------------------------------------------------- */

int FixSurfaceLocal::overlap_bins_3d(double *pt, double eps, int *indices)
{
  int ilo = static_cast<int> ((pt[0]-eps-bboxlo[0]) * invbinx);
  int ihi = static_cast<int> ((pt[0]+eps-bboxlo[0]) * invbinx);
  int jlo = static_cast<int> ((pt[1]-eps-bboxlo[1]) * invbiny);
  int jhi = static_cast<int> ((pt[1]+eps-bboxlo[1]) * invbiny);
  int klo = static_cast<int> ((pt[2]-eps-bboxlo[2]) * invbinz);
  int khi = static_cast<int> ((pt[2]+eps-bboxlo[2]) * invbinz);
  
  ilo = MAX(ilo,0);
  ihi = MIN(ihi,nbinx-1);
  jlo = MAX(jlo,0);
  jhi = MIN(jhi,nbiny-1);
  klo = MAX(klo,0);
  khi = MIN(khi,nbinz-1);

  if (ilo == ihi && jlo == jhi && klo == khi) {
    indices[0] = klo*nbiny*nbinx + jlo*nbinx + ilo;
    return 1;
  }

  int i,j,k;
  int n = 0;
  for (k = klo; k <= khi; k++)
    for (j = jlo; j <= jhi; j++)
      for (i = ilo; i <= ihi; i++)
        indices[n++] = k*nbiny*nbinx + j*nbinx + i;
  return n;
}

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// methods for global surf connectivity build, assignment to procs
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------

/* ----------------------------------------------------------------------
   check if distributed line/tri particles already exist
------------------------------------------------------------------------- */

int FixSurfaceLocal::check_exist()
{
  int *surf = atom->line;
  if (dimension == 3) surf = atom->tri;
  int nlocal = atom->nlocal;

  int flag = 0;
  for (int i = 0; i < nlocal; i++)
    if (surf[i] >= 0) flag = 1;

  int flagall;
  MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_MAX,world);

  if (flagall) return 1;
  return 0;
}

/* ----------------------------------------------------------------------
   extract lines or surfs from molecule template ID for one or more molecules
   concatenate into single list of lines and tris
   create list of unique points using hash
------------------------------------------------------------------------- */

void FixSurfaceLocal::extract_from_molecules(char *molID)
{
  // populate global point/line/tri data structs

  npoints = nlines = ntris = 0;
  int maxpoints = 0;

  int imol = atom->find_molecule(molID);
  if (imol == -1)
    error->all(FLERR,"Molecule template ID for fix surface/local does not exist");

  // loop over one or more molecules in molID

  Molecule **onemols = &atom->molecules[imol];
  int nmol = onemols[0]->nset;

  for (int m = 0; m < nmol; m++) {
    if (dimension == 2)
      if (onemols[m]->lineflag == 0)
        error->all(FLERR,"Fix surface/local molecule must have lines");
    if (dimension == 3)
      if (onemols[m]->triflag == 0)
        error->all(FLERR,"Fix surface/local molecule must have triangles");

    int nl = onemols[m]->nlines;
    int nt = onemols[m]->ntris;

    nlines += nl;
    ntris += nt;
    lines = (Line *) memory->srealloc(lines,nlines*sizeof(Line),
                                      "surface/global:lines");
    tris = (Tri *) memory->srealloc(tris,ntris*sizeof(Tri),
                                    "surface/global:tris");

    // create a map
    // key = xyz coords of a point
    // value = index into unique points vector

    std::map<std::tuple<double,double,double>,int> hash;

    // offset line/tri index lists by previous npoints
    // pi,p2,p3 are C-style indices into points vector

    if (dimension == 2) {
      int *molline = onemols[m]->molline;
      int *typeline = onemols[m]->typeline;
      double **epts = onemols[m]->lines;
      int iline = nlines - nl;

      for (int i = 0; i < nl; i++) {
        lines[iline].mol = molline[i];
        lines[iline].type = typeline[i];

        auto key = std::make_tuple(epts[i][0],epts[i][1],0.0);
        if (hash.find(key) == hash.end()) {
          if (npoints == maxpoints) {
            maxpoints += DELTA;
            points = (Point *) memory->srealloc(points,maxpoints*sizeof(Point),
                                                "surface/local:points");
          }
          hash[key] = npoints;
          points[npoints].x[0] = epts[i][0];
          points[npoints].x[1] = epts[i][1];
          points[npoints].x[2] = 0.0;
          lines[iline].p1 = npoints;
          npoints++;
        } else lines[iline].p1 = hash[key];

        key = std::make_tuple(epts[i][2],epts[i][3],0.0);
        if (hash.find(key) == hash.end()) {
          if (npoints == maxpoints) {
            maxpoints += DELTA;
            points = (Point *) memory->srealloc(points,maxpoints*sizeof(Point),
                                                "surface/local:points");
          }
          hash[key] = npoints;
          points[npoints].x[0] = epts[i][2];
          points[npoints].x[1] = epts[i][3];
          points[npoints].x[2] = 0.0;
          lines[iline].p2 = npoints;
          npoints++;
        } else lines[iline].p2 = hash[key];

        iline++;
      }
    }

    if (dimension == 3) {
      int *moltri = onemols[m]->moltri;
      int *typetri = onemols[m]->typetri;
      double **cpts = onemols[m]->tris;
      int itri = ntris - nt;

      for (int i = 0; i < nt; i++) {
        tris[itri].mol = moltri[i];
        tris[itri].type = typetri[i];

        auto key = std::make_tuple(cpts[i][0],cpts[i][1],cpts[i][2]);
        if (hash.find(key) == hash.end()) {
          if (npoints == maxpoints) {
            maxpoints += DELTA;
            points = (Point *) memory->srealloc(points,maxpoints*sizeof(Point),
                                                "surface/local:points");
          }
          hash[key] = npoints;
          points[npoints].x[0] = cpts[i][0];
          points[npoints].x[1] = cpts[i][1];
          points[npoints].x[2] = cpts[i][2];
          tris[itri].p1 = npoints;
          npoints++;
        } else tris[itri].p1 = hash[key];

        key = std::make_tuple(cpts[i][3],cpts[i][4],cpts[i][5]);
        if (hash.find(key) == hash.end()) {
          if (npoints == maxpoints) {
            maxpoints += DELTA;
            points = (Point *) memory->srealloc(points,maxpoints*sizeof(Point),
                                                "surface/local:points");
          }
          hash[key] = npoints;
          points[npoints].x[0] = cpts[i][3];
          points[npoints].x[1] = cpts[i][4];
          points[npoints].x[2] = cpts[i][5];
          tris[itri].p2 = npoints;
          npoints++;
        } else tris[itri].p2 = hash[key];

        key = std::make_tuple(cpts[i][6],cpts[i][7],cpts[i][8]);
        if (hash.find(key) == hash.end()) {
          if (npoints == maxpoints) {
            maxpoints += DELTA;
            points = (Point *) memory->srealloc(points,maxpoints*sizeof(Point),
                                                "surface/local:points");
          }
          hash[key] = npoints;
          points[npoints].x[0] = cpts[i][6];
          points[npoints].x[1] = cpts[i][7];
          points[npoints].x[2] = cpts[i][8];
          tris[itri].p3 = npoints;
          npoints++;
        } else tris[itri].p3 = hash[key];

        itri++;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   extract triangles from an STL file, can be text or binary
   create list of unique points using hash
------------------------------------------------------------------------- */

void FixSurfaceLocal::extract_from_stlfile(char *filename)
{
  if (dimension == 2)
    error->all(FLERR,"Fix surface/global cannot use an STL file for 2d simulations");

  // read tris from STL file
  // stltris = tri coords internal to STL reader

  STLReader *stl = new STLReader(lmp);
  double **stltris;
  ntris = stl->read_file(filename,stltris);

  // create points and tris data structs

  npoints = 0;
  int maxpoints = 0;

  tris = (Tri *) memory->smalloc(ntris*sizeof(Tri),"surface/global:tris");

  // create a map
  // key = xyz coords of a point
  // value = index into unique points vector

  std::map<std::tuple<double,double,double>,int> hash;

  // loop over STL tris
  // populate points and tris data structs
  // set molecule and type of tri = 1

  for (int itri = 0; itri < ntris; itri++) {
    tris[itri].mol = 1;
    tris[itri].type = 1;

    auto key = std::make_tuple(stltris[itri][0],stltris[itri][1],stltris[itri][2]);
    if (hash.find(key) == hash.end()) {
      if (npoints == maxpoints) {
        maxpoints += DELTA;
        points = (Point *) memory->srealloc(points,maxpoints*sizeof(Point),
                                            "surface/global:points");
      }
      hash[key] = npoints;
      points[npoints].x[0] = stltris[itri][0];
      points[npoints].x[1] = stltris[itri][1];
      points[npoints].x[2] = stltris[itri][2];
      tris[itri].p1 = npoints;
      npoints++;
    } else tris[itri].p1 = hash[key];

    key = std::make_tuple(stltris[itri][3],stltris[itri][4],stltris[itri][5]);
    if (hash.find(key) == hash.end()) {
      if (npoints == maxpoints) {
        maxpoints += DELTA;
        points = (Point *) memory->srealloc(points,maxpoints*sizeof(Point),
                                            "surface/global:points");
      }
      hash[key] = npoints;
      points[npoints].x[0] = stltris[itri][3];
      points[npoints].x[1] = stltris[itri][4];
      points[npoints].x[2] = stltris[itri][5];
      tris[itri].p2 = npoints;
      npoints++;
    } else tris[itri].p2 = hash[key];

    key = std::make_tuple(stltris[itri][6],stltris[itri][7],stltris[itri][8]);
    if (hash.find(key) == hash.end()) {
      if (npoints == maxpoints) {
        maxpoints += DELTA;
        points = (Point *) memory->srealloc(points,maxpoints*sizeof(Point),
                                            "surface/global:points");
      }
      hash[key] = npoints;
      points[npoints].x[0] = stltris[itri][6];
      points[npoints].x[1] = stltris[itri][7];
      points[npoints].x[2] = stltris[itri][8];
      tris[itri].p3 = npoints;
      npoints++;
    } else tris[itri].p3 = hash[key];
  }

  // delete STL reader

  delete stl;
}

/* ----------------------------------------------------------------------
   create and initialize partial Connect2d info for all lines
   this can be done with EXACT point matching
     since global points were inferred from molecule files
     and all procs store a copy of all global points
   info is stored here in connect2dall
     will be stored in connect2d in assign2d()
     remainder will be initialized in connect2d_complete()
------------------------------------------------------------------------- */

void FixSurfaceLocal::connectivity2d_global()
{
  // allocate and initilize global connectivity list
  // connect2atom will be initialized when assign to procs

  connect2dall = (Connect2d *) memory->smalloc(nlines*sizeof(Connect2d),
                                               "surface/local:connect2dall");

  // setup line end point connectivity lists
  // counts = # of lines containing each end point (including self)
  // plines = ragged 2d array with indices of lines which contain each point

  int *counts;
  memory->create(counts,npoints,"surface/local:count");

  for (int i = 0; i < npoints; i++) counts[i] = 0;

  for (int i = 0; i < nlines; i++) {
    counts[lines[i].p1]++;
    counts[lines[i].p2]++;
  }

  int **plines;
  memory->create_ragged(plines,npoints,counts,"surface/local:plines");

  for (int i = 0; i < npoints; i++) counts[i] = 0;

  for (int i = 0; i < nlines; i++) {
    plines[lines[i].p1][counts[lines[i].p1]++] = i;
    plines[lines[i].p2][counts[lines[i].p2]++] = i;
  }

  // p12_counts = # of lines connecting to endpoints p12 of each line
  // do NOT include self

  int *p1_counts,*p2_counts;
  memory->create(p1_counts,nlines,"surface/local:p1_counts");
  memory->create(p2_counts,nlines,"surface/local:p2_counts");

  for (int i = 0; i < nlines; i++) {
    p1_counts[i] = counts[lines[i].p1] - 1;
    p2_counts[i] = counts[lines[i].p2] - 1;
  }

  // allocate all ragged arrays which connect2dall will point to

  memory->create_ragged(neigh_p1,nlines,p1_counts,"surface/local:neigh_p1");
  memory->create_ragged(neigh_p2,nlines,p2_counts,"surface/local:neigh_p2");

  // set connect2dall vector ptrs to rows of corresponding ragged arrays

  for (int i = 0; i < nlines; i++) {
    connect2dall[i].np1 = p1_counts[i];
    if (connect2dall[i].np1 == 0) connect2dall[i].neigh_p1 = nullptr;
    else connect2dall[i].neigh_p1 = neigh_p1[i];

    connect2dall[i].np2 = p2_counts[i];
    if (connect2dall[i].np2 == 0) connect2dall[i].neigh_p2 = nullptr;
    else connect2dall[i].neigh_p2 = neigh_p2[i];
  }

  // initialize connect2dall neigh vectors for each end point of each line
  //   do NOT include self
  // in connect2dall, neigh_p12 stores global line indices (0 to Nline-1)
  // in final connect2d, neigh_p12 will store line IDs

  int j,m;

  for (int i = 0; i < nlines; i++) {
    if (p1_counts[i]) {
      j = 0;
      for (m = 0; m <= p1_counts[i]; m++) {
        if (plines[lines[i].p1][m] == i) continue;
        connect2dall[i].neigh_p1[j] = plines[lines[i].p1][m];
        j++;
      }
    }
    if (p2_counts[i]) {
      j = 0;
      for (m = 0; m <= p2_counts[i]; m++) {
        if (plines[lines[i].p2][m] == i) continue;
        connect2dall[i].neigh_p2[j] = plines[lines[i].p2][m];
        j++;
      }
    }
  }

  // deallocate counts, plines, p12_counts

  memory->destroy(counts);
  memory->destroy(plines);
  memory->destroy(p1_counts);
  memory->destroy(p2_counts);
}

/* ----------------------------------------------------------------------
   create and initialize partial Connect3d info for all tris
   this can be done with EXACT point matching
     since global points were inferred from molecule or STL files
     and all procs store a copy of all global points
   info is stored here in connect3dall
     will be stored in connect3d in assign3d()
     remainder will be initialized in connect3d_complete()
------------------------------------------------------------------------- */

void FixSurfaceLocal::connectivity3d_global()
{
  int p1,p2,p3;

  // allocate and initilize global connectivity list
  // connect2atom will be initialized when assign to procs

  connect3dall = (Connect3d *) memory->smalloc(ntris*sizeof(Connect3d),
                                               "surface/local:connect3dall");

  // create hash = map of unique edges
  //   key = <p1,p2> indices of 2 points, in either order
  //   value = index of the unique edge (0 to Nedge-1)
  // tri2edges[i][j] = index of unique edge for tri I and edge J
  // nedges = total count of unique edges

  int **tri2edge;
  memory->create(tri2edge,ntris,3,"surfface/global::tri2edge");

  std::map<std::tuple<int,int>,int> hash;
  int nedges = 0;

  for (int i = 0; i < ntris; i++) {
    p1 = tris[i].p1;
    p2 = tris[i].p2;
    p3 = tris[i].p3;

    auto key1 = std::make_tuple(p1,p2);
    auto key2 = std::make_tuple(p2,p1);

    if (hash.find(key1) == hash.end() && hash.find(key2) == hash.end()) {
      hash[key1] = nedges;
      tri2edge[i][0] = nedges;
      nedges++;
    }
    else if (hash.find(key1) != hash.end()) tri2edge[i][0] = hash[key1];
    else if (hash.find(key2) != hash.end()) tri2edge[i][0] = hash[key2];

    key1 = std::make_tuple(p2,p3);
    key2 = std::make_tuple(p3,p2);

    if (hash.find(key1) == hash.end() && hash.find(key2) == hash.end()) {
      hash[key1] = nedges;
      tri2edge[i][1] = nedges;
      nedges++;
    }
    else if (hash.find(key1) != hash.end()) tri2edge[i][1] = hash[key1];
    else if (hash.find(key2) != hash.end()) tri2edge[i][1] = hash[key2];

    key1 = std::make_tuple(p3,p1);
    key2 = std::make_tuple(p1,p3);

    if (hash.find(key1) == hash.end() && hash.find(key2) == hash.end()) {
      hash[key1] = nedges;
      tri2edge[i][2] = nedges;
      nedges++;
    }
    else if (hash.find(key1) != hash.end()) tri2edge[i][2] = hash[key1];
    else if (hash.find(key2) != hash.end()) tri2edge[i][2] = hash[key2];
  }

  // setup tri edge connectivity lists
  // counts = # of tris containing each edge (including self)
  // etris = ragged 2d array with indices of tris which contain each edge

  int *counts;
  memory->create(counts,nedges,"surface/local:count");

  for (int i = 0; i < nedges; i++) counts[i] = 0;

  for (int i = 0; i < ntris; i++) {
    counts[tri2edge[i][0]]++;
    counts[tri2edge[i][1]]++;
    counts[tri2edge[i][2]]++;
  }

  int **etris;
  memory->create_ragged(etris,nedges,counts,"surface/local:etris");

  for (int i = 0; i < nedges; i++) counts[i] = 0;

  for (int i = 0; i < ntris; i++) {
    etris[tri2edge[i][0]][counts[tri2edge[i][0]]++] = i;
    etris[tri2edge[i][1]][counts[tri2edge[i][1]]++] = i;
    etris[tri2edge[i][2]][counts[tri2edge[i][2]]++] = i;
  }

  // e123_counts = # of edges connecting to edges e123 of each tri
  // do NOT include self

  int *e1_counts,*e2_counts,*e3_counts;
  memory->create(e1_counts,ntris,"surface/local:e1_counts");
  memory->create(e2_counts,ntris,"surface/local:e2_counts");
  memory->create(e3_counts,ntris,"surface/local:e3_counts");

  for (int i = 0; i < ntris; i++) {
    e1_counts[i] = counts[tri2edge[i][0]] - 1;
    e2_counts[i] = counts[tri2edge[i][1]] - 1;
    e3_counts[i] = counts[tri2edge[i][2]] - 1;
  }

  // allocate all edge ragged arrays which connect3dall will point to

  memory->create_ragged(neigh_e1,ntris,e1_counts,"surface/global:neigh_e1");
  memory->create_ragged(neigh_e2,ntris,e2_counts,"surface/global:neigh_e2");
  memory->create_ragged(neigh_e3,ntris,e3_counts,"surface/global:neigh_e3");
  
  // set connectedall vector ptrs to rows of corresponding ragged arrays

  for (int i = 0; i < ntris; i++) {
    connect3dall[i].ne1 = e1_counts[i];
    if (connect3dall[i].ne1) connect3dall[i].neigh_e1 = neigh_e1[i];
    else connect3dall[i].neigh_e1 = nullptr;
    connect3dall[i].ne2 = e2_counts[i];
    if (connect3dall[i].ne2) connect3dall[i].neigh_e2 = neigh_e2[i];
    else connect3dall[i].neigh_e1 = nullptr;
    connect3dall[i].ne3 = e3_counts[i];
    if (connect3dall[i].ne3) connect3dall[i].neigh_e3 = neigh_e3[i];
    else connect3dall[i].neigh_e3 = nullptr;
  }

  // initialize connect3dall edge neigh vectors for each edge of each tri
  //   do NOT include self
  // in connecte3dall, neigh_e123 stores global tri indices (0 to Ntri-1)
  // in final connect3d, neigh_e123 will store tri IDs

  int j,m;

  for (int i = 0; i < ntris; i++) {
    if (connect3dall[i].ne1) {
      j = 0;
      for (m = 0; m < counts[tri2edge[i][0]]; m++) {
        if (etris[tri2edge[i][0]][m] == i) continue;
        connect3dall[i].neigh_e1[j] = etris[tri2edge[i][0]][m];
        j++;
      }
    }
    if (connect3dall[i].ne2) {
      j = 0;
      for (m = 0; m < counts[tri2edge[i][1]]; m++) {
        if (etris[tri2edge[i][1]][m] == i) continue;
        connect3dall[i].neigh_e2[j] = etris[tri2edge[i][1]][m];
        j++;
      }
    }
    if (connect3dall[i].ne3) {
      j = 0;
      for (m = 0; m < counts[tri2edge[i][2]]; m++) {
        if (etris[tri2edge[i][2]][m] == i) continue;
        connect3dall[i].neigh_e3[j] = etris[tri2edge[i][2]][m];
        j++;
      }
    }
  }
  
  memory->destroy(counts);
  memory->destroy(tri2edge);
  memory->destroy(etris);
  memory->destroy(e1_counts);
  memory->destroy(e2_counts);
  memory->destroy(e3_counts);

  // setup corner point connectivity lists
  // counts = # of tris containing each point (including self)
  // ctris = ragged 2d array with indices of tris which contain each point

  memory->create(counts,npoints,"surface/local:count");

  for (int i = 0; i < npoints; i++) counts[i] = 0;

  for (int i = 0; i < ntris; i++) {
    counts[tris[i].p1]++;
    counts[tris[i].p2]++;
    counts[tris[i].p3]++;
  }

  int **ctris;
  memory->create_ragged(ctris,npoints,counts,"surface/local:ctris");

  for (int i = 0; i < npoints; i++) counts[i] = 0;

  for (int i = 0; i < ntris; i++) {
    ctris[tris[i].p1][counts[tris[i].p1]++] = i;
    ctris[tris[i].p2][counts[tris[i].p2]++] = i;
    ctris[tris[i].p3][counts[tris[i].p3]++] = i;
  }

  // c123_counts = # of tris connecting to corners c123 of each tri
  // do NOT include self or tris which connect to an edge

  int *c1_counts,*c2_counts,*c3_counts;
  memory->create(c1_counts,ntris,"surface/local:c1_counts");
  memory->create(c2_counts,ntris,"surface/local:c2_counts");
  memory->create(c3_counts,ntris,"surface/local:c3_counts");

  for (int i = 0; i < ntris; i++) {
    c1_counts[i] = counts[tris[i].p1] - 1;
    c1_counts[i] -= connect3dall[i].ne3 + connect3dall[i].ne1;
    c2_counts[i] = counts[tris[i].p2] - 1;
    c2_counts[i] -= connect3dall[i].ne1 + connect3dall[i].ne2;
    c3_counts[i] = counts[tris[i].p3] - 1;
    c3_counts[i] -= connect3dall[i].ne2 + connect3dall[i].ne3;
  }

  // allocate all corner ragged arrays which connect3dall will point to

  memory->create_ragged(neigh_c1,ntris,c1_counts,"surface/global:neigh_c1");
  memory->create_ragged(neigh_c2,ntris,c2_counts,"surface/global:neigh_c2");
  memory->create_ragged(neigh_c3,ntris,c3_counts,"surface/global:neigh_c3");

  // set connect3dall corner vector ptrs to rows of corresponding corner ragged arrays

  for (int i = 0; i < ntris; i++) {
    connect3dall[i].nc1 = c1_counts[i];
    if (connect3dall[i].nc1) connect3dall[i].neigh_c1 = neigh_c1[i];
    else connect3dall[i].neigh_c1 = nullptr;

    connect3dall[i].nc2 = c2_counts[i];
    if (connect3dall[i].nc2) connect3dall[i].neigh_c2 = neigh_c2[i];
    else connect3dall[i].neigh_c2 = nullptr;

    connect3dall[i].nc3 = c3_counts[i];
    if (connect3dall[i].nc3) connect3dall[i].neigh_c3 = neigh_c3[i];
    else connect3dall[i].neigh_c3 = nullptr;
  }

  // initialize connect3dall corner neigh vectors for each corner of each tri
  // do NOT include self or tris which connect to an edge
  // only include tris which only connect at the corner point
  // in connect3dall, neigh_c123 stores global tri indices (0 to Ntri-1)
  // in final connect3d, neigh_c123 will store tri IDs

  int n,medge,skipflag;

  for (int i = 0; i < ntris; i++) {
    if (connect3dall[i].nc1) {
      j = 0;
      for (m = 0; m < counts[tris[i].p1]; m++) {
        n = ctris[tris[i].p1][m];
        if (n == i) continue;

        skipflag = 0;
        for (medge = 0; medge < connect3dall[i].ne3; medge++)
          if (n == connect3dall[i].neigh_e3[medge]) skipflag = 1;
        for (medge = 0; medge < connect3dall[i].ne1; medge++)
          if (n == connect3dall[i].neigh_e1[medge]) skipflag = 1;
        if (skipflag) continue;

        connect3dall[i].neigh_c1[j] = n;
        j++;
      }
    }
    if (connect3dall[i].nc2) {
      j = 0;
      for (m = 0; m < counts[tris[i].p2]; m++) {
        n = ctris[tris[i].p2][m];
        if (n == i) continue;

        skipflag = 0;
        for (medge = 0; medge < connect3dall[i].ne1; medge++)
          if (n == connect3dall[i].neigh_e1[medge]) skipflag = 1;
        for (medge = 0; medge < connect3dall[i].ne2; medge++)
          if (n == connect3dall[i].neigh_e2[medge]) skipflag = 1;
        if (skipflag) continue;

        connect3dall[i].neigh_c2[j] = n;
        j++;
      }
    }
    if (connect3dall[i].nc3) {
      j = 0;
      for (m = 0; m < counts[tris[i].p3]; m++) {
        n = ctris[tris[i].p3][m];
        if (n == i) continue;

        skipflag = 0;
        for (medge = 0; medge < connect3dall[i].ne2; medge++)
          if (n == connect3dall[i].neigh_e2[medge]) skipflag = 1;
        for (medge = 0; medge < connect3dall[i].ne3; medge++)
          if (n == connect3dall[i].neigh_e3[medge]) skipflag = 1;
        if (skipflag) continue;

        connect3dall[i].neigh_c3[j] = n;
        j++;
      }
    }
  }
  
  // clean up
  
  memory->destroy(counts);
  memory->destroy(ctris);
  memory->destroy(c1_counts);
  memory->destroy(c2_counts);
  memory->destroy(c3_counts);
}

/* ----------------------------------------------------------------------
   assign lines and their connectivity from global data structs to each proc
   logic for proc assign follows Atom::data_atoms(), as if read from data file
------------------------------------------------------------------------- */

void FixSurfaceLocal::assign2d()
{
  AtomVec *avec = atom->avec;
  avec_line = (AtomVecLine *) atom->style_match("line");
  if (!avec_line)
    error->all(FLERR,"Fix surface/local requires atom style line");

  // set bounds for my proc
  // if periodic and I am lo/hi proc, adjust bounds by EPSILON
  // insures all data atoms will be owned even with round-off

  int triclinic = domain->triclinic;

  double epsilon[3];
  if (triclinic) epsilon[0] = epsilon[1] = epsilon[2] = EPSILON;
  else {
    epsilon[0] = domain->prd[0] * EPSILON;
    epsilon[1] = domain->prd[1] * EPSILON;
    epsilon[2] = domain->prd[2] * EPSILON;
  }

  double sublo[3],subhi[3];
  if (triclinic == 0) {
    sublo[0] = domain->sublo[0]; subhi[0] = domain->subhi[0];
    sublo[1] = domain->sublo[1]; subhi[1] = domain->subhi[1];
    sublo[2] = domain->sublo[2]; subhi[2] = domain->subhi[2];
  } else {
    sublo[0] = domain->sublo_lamda[0]; subhi[0] = domain->subhi_lamda[0];
    sublo[1] = domain->sublo_lamda[1]; subhi[1] = domain->subhi_lamda[1];
    sublo[2] = domain->sublo_lamda[2]; subhi[2] = domain->subhi_lamda[2];
  }

  if (comm->layout != LAYOUT_TILED) {
    if (domain->xperiodic) {
      if (comm->myloc[0] == 0) sublo[0] -= epsilon[0];
      if (comm->myloc[0] == comm->procgrid[0]-1) subhi[0] += epsilon[0];
    }
    if (domain->yperiodic) {
      if (comm->myloc[1] == 0) sublo[1] -= epsilon[1];
      if (comm->myloc[1] == comm->procgrid[1]-1) subhi[1] += epsilon[1];
    }
    if (domain->zperiodic) {
      if (comm->myloc[2] == 0) sublo[2] -= epsilon[2];
      if (comm->myloc[2] == comm->procgrid[2]-1) subhi[2] += epsilon[2];
    }

  } else {
    if (domain->xperiodic) {
      if (comm->mysplit[0][0] == 0.0) sublo[0] -= epsilon[0];
      if (comm->mysplit[0][1] == 1.0) subhi[0] += epsilon[0];
    }
    if (domain->yperiodic) {
      if (comm->mysplit[1][0] == 0.0) sublo[1] -= epsilon[1];
      if (comm->mysplit[1][1] == 1.0) subhi[1] += epsilon[1];
    }
    if (domain->zperiodic) {
      if (comm->mysplit[2][0] == 0.0) sublo[2] -= epsilon[2];
      if (comm->mysplit[2][1] == 1.0) subhi[2] += epsilon[2];
    }
  }

  // idmaxall = largest existing atom ID
  // set atom2connect = -1 for all current particles

  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;

  tagint idmax = 0;
  for (int i = 0; i < nlocal; i++) {
    idmax = MAX(tag[i],idmax);
    atom2connect[i] = -1;
  }
  tagint idmaxall;
  MPI_Allreduce(&idmax,&idmaxall,1,MPI_LMP_TAGINT,MPI_MAX,world);

  // loop over all global lines
  // compute their midpoints
  // keep lines belonging to this proc

  int j,n,num;
  imageint imagedata;
  double xmid[3],lamda[3];
  double *coord,*x1,*x2;
  tagint *global,*local;

  char pts[4][32];
  char *values[4];
  for (int i = 0; i < 4; i++) values[i] = &pts[i][0];
  std::vector<std::string> svalues(5);

  for (int i = 0; i < nlines; i++) {

    // midpt of line

    x1 = points[lines[i].p1].x;
    x2 = points[lines[i].p2].x;
    xmid[0] = 0.5*(x1[0]+x2[0]);
    xmid[1] = 0.5*(x1[1]+x2[1]);
    xmid[2] = 0.5*(x1[2]+x2[2]);

    imagedata = ((imageint) IMGMAX << IMG2BITS) |
      ((imageint) IMGMAX << IMGBITS) | IMGMAX;

    domain->remap(xmid,imagedata);
    if (triclinic) {
      domain->x2lamda(xmid,lamda);
      coord = lamda;
    } else coord = xmid;

    if (coord[0] >= sublo[0] && coord[0] < subhi[0] &&
        coord[1] >= sublo[1] && coord[1] < subhi[1] &&
        coord[2] >= sublo[2] && coord[2] < subhi[2]) {

      // create a default particle of correct style

      avec->create_atom(lines[i].type,xmid);

      // change it to be a line

      int n = atom->nlocal - 1;
      atom->tag[n] = idmaxall + i + 1;
      atom->image[n] = imagedata;
      atom->molecule[n] = lines[i].mol;
      atom->line[n] = 0;
      atom->rmass[n] = 1.0;              // set line density = 1.0
      atom->radius[n] = 0.0;

      // set the bonus data for a line = end points

      sprintf(values[0],"%-1.16e",x1[0]);
      sprintf(values[1],"%-1.16e",x1[1]);
      sprintf(values[2],"%-1.16e",x2[0]);
      sprintf(values[3],"%-1.16e",x2[1]);

      // svalues[0] is not used here
      // used when caller of data_atom_bonus() is via read_data command

      svalues[1] = (const char *) values[0];
      svalues[2] = (const char *) values[1];
      svalues[3] = (const char *) values[2];
      svalues[4] = (const char *) values[3];

      avec_line->data_atom_bonus(n,svalues);

      // allocate all vectors in Connect2d
      // copy neigh_p12 from global connect2dall to local connect2d
      //   reset global line indices to new line IDs beyond idmaxall
      // other vectors will be set in connectivity2d_complete()

      if (nlocal_connect == nmax_connect) grow_connect();
      memcpy(&connect2d[nlocal_connect],&connect2dall[i],sizeof(Connect2d));

      num = connect2d[nlocal_connect].np1;
      if (num) {
        connect2d[nlocal_connect].neigh_p1 = tcp->get(num,pool2d[nlocal_connect].neigh_p1);
        connect2d[nlocal_connect].pwhich_p1 = tcp->get(num,pool2d[nlocal_connect].pwhich_p1);
        connect2d[nlocal_connect].nside_p1 = tcp->get(num,pool2d[nlocal_connect].nside_p1);
        connect2d[nlocal_connect].aflag_p1 = tcp->get(num,pool2d[nlocal_connect].aflag_p1);
        global = connect2dall[nlocal_connect].neigh_p1;
        local = connect2d[nlocal_connect].neigh_p1;
        for (j = 0; j < num; j++)
          local[j] = global[j] + idmaxall + 1;
      } else {
        connect2d[nlocal_connect].neigh_p1 = nullptr;
        connect2d[nlocal_connect].pwhich_p1 = nullptr;
        connect2d[nlocal_connect].nside_p1 = nullptr;
        connect2d[nlocal_connect].aflag_p1 = nullptr;
      }
      
      num = connect2d[nlocal_connect].np2;
      if (num) {
        connect2d[nlocal_connect].neigh_p2 = tcp->get(num,pool2d[nlocal_connect].neigh_p2);
        connect2d[nlocal_connect].pwhich_p2 = tcp->get(num,pool2d[nlocal_connect].pwhich_p2);
        connect2d[nlocal_connect].nside_p2 = tcp->get(num,pool2d[nlocal_connect].nside_p2);
        connect2d[nlocal_connect].aflag_p2 = tcp->get(num,pool2d[nlocal_connect].aflag_p2);
        global = connect2dall[nlocal_connect].neigh_p2;
        local = connect2d[nlocal_connect].neigh_p2;
        for (j = 0; j < num; j++)
          local[j] = global[j] + idmaxall + 1;
      } else {
        connect2d[nlocal_connect].neigh_p2 = nullptr;
        connect2d[nlocal_connect].pwhich_p2 = nullptr;
        connect2d[nlocal_connect].nside_p2 = nullptr;
        connect2d[nlocal_connect].aflag_p2 = nullptr;
      }
      
      connect2atom[nlocal_connect] = n;
      atom2connect[n] = nlocal_connect;
      nlocal_connect++;
    }
  }

  memory->sfree(connect2dall);
  memory->destroy(neigh_p1);
  memory->destroy(neigh_p2);

  // recreate atom map that includes added lines

  if (atom->map_style) {
    atom->map_init();
    atom->map_set();
  }
}

/* ----------------------------------------------------------------------
   assign tris and their connectivity from global data structs to each proc
------------------------------------------------------------------------- */

void FixSurfaceLocal::assign3d()
{
  AtomVec *avec = atom->avec;
  avec_tri = (AtomVecTri *) atom->style_match("tri");
  if (!avec_tri)
    error->all(FLERR,"Fix surface/local requires atom style tri");

  // set bounds for my proc
  // if periodic and I am lo/hi proc, adjust bounds by EPSILON
  // insures all data atoms will be owned even with round-off

  int triclinic = domain->triclinic;

  double epsilon[3];
  if (triclinic) epsilon[0] = epsilon[1] = epsilon[2] = EPSILON;
  else {
    epsilon[0] = domain->prd[0] * EPSILON;
    epsilon[1] = domain->prd[1] * EPSILON;
    epsilon[2] = domain->prd[2] * EPSILON;
  }

  double sublo[3],subhi[3];
  if (triclinic == 0) {
    sublo[0] = domain->sublo[0]; subhi[0] = domain->subhi[0];
    sublo[1] = domain->sublo[1]; subhi[1] = domain->subhi[1];
    sublo[2] = domain->sublo[2]; subhi[2] = domain->subhi[2];
  } else {
    sublo[0] = domain->sublo_lamda[0]; subhi[0] = domain->subhi_lamda[0];
    sublo[1] = domain->sublo_lamda[1]; subhi[1] = domain->subhi_lamda[1];
    sublo[2] = domain->sublo_lamda[2]; subhi[2] = domain->subhi_lamda[2];
  }

  if (comm->layout != LAYOUT_TILED) {
    if (domain->xperiodic) {
      if (comm->myloc[0] == 0) sublo[0] -= epsilon[0];
      if (comm->myloc[0] == comm->procgrid[0]-1) subhi[0] += epsilon[0];
    }
    if (domain->yperiodic) {
      if (comm->myloc[1] == 0) sublo[1] -= epsilon[1];
      if (comm->myloc[1] == comm->procgrid[1]-1) subhi[1] += epsilon[1];
    }
    if (domain->zperiodic) {
      if (comm->myloc[2] == 0) sublo[2] -= epsilon[2];
      if (comm->myloc[2] == comm->procgrid[2]-1) subhi[2] += epsilon[2];
    }

  } else {
    if (domain->xperiodic) {
      if (comm->mysplit[0][0] == 0.0) sublo[0] -= epsilon[0];
      if (comm->mysplit[0][1] == 1.0) subhi[0] += epsilon[0];
    }
    if (domain->yperiodic) {
      if (comm->mysplit[1][0] == 0.0) sublo[1] -= epsilon[1];
      if (comm->mysplit[1][1] == 1.0) subhi[1] += epsilon[1];
    }
    if (domain->zperiodic) {
      if (comm->mysplit[2][0] == 0.0) sublo[2] -= epsilon[2];
      if (comm->mysplit[2][1] == 1.0) subhi[2] += epsilon[2];
    }
  }

  // idmaxall = largest existing atom ID
  // set atom2connect = -1 for all current particles

  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;

  tagint idmax = 0;
  for (int i = 0; i < nlocal; i++) {
    idmax = MAX(tag[i],idmax);
    atom2connect[i] = -1;
  }
  tagint idmaxall;
  MPI_Allreduce(&idmax,&idmaxall,1,MPI_LMP_TAGINT,MPI_MAX,world);

  // loop over all global triangles
  // compute their center points
  // keep lines belonging to this proc

  int j,n,num;
  imageint imagedata;
  double xmid[3],lamda[3];
  double *coord,*x1,*x2,*x3;
  tagint *global,*local;

  char pts[9][32];
  char *values[9];
  for (int i = 0; i < 9; i++) values[i] = &pts[i][0];
  std::vector<std::string> svalues(10);

  for (int i = 0; i < ntris; i++) {

    // center pt of triangle

    x1 = points[tris[i].p1].x;
    x2 = points[tris[i].p2].x;
    x3 = points[tris[i].p3].x;
    xmid[0] = (x1[0]+x2[0]+x3[0]) / 3.0;
    xmid[1] = (x1[1]+x2[1]+x3[1]) / 3.0;
    xmid[2] = (x1[2]+x2[2]+x3[2]) / 3.0;

    imagedata = ((imageint) IMGMAX << IMG2BITS) |
      ((imageint) IMGMAX << IMGBITS) | IMGMAX;

    domain->remap(xmid,imagedata);
    if (triclinic) {
      domain->x2lamda(xmid,lamda);
      coord = lamda;
    } else coord = xmid;

    if (coord[0] >= sublo[0] && coord[0] < subhi[0] &&
        coord[1] >= sublo[1] && coord[1] < subhi[1] &&
        coord[2] >= sublo[2] && coord[2] < subhi[2]) {

      // create a default particle of correct style

      avec->create_atom(tris[i].type,xmid);

      // change it to be a triangle

      int n = atom->nlocal - 1;
      atom->tag[n] = idmaxall + i + 1;
      atom->image[n] = imagedata;
      atom->molecule[n] = tris[i].mol;
      atom->tri[n] = 0;
      atom->rmass[n] = 1.0;              // set tri density = 1.0
      atom->radius[n] = 0.0;

      // set the bonus data for a tri = corner pts

      sprintf(values[0],"%-1.16e",x1[0]);
      sprintf(values[1],"%-1.16e",x1[1]);
      sprintf(values[2],"%-1.16e",x1[2]);
      sprintf(values[3],"%-1.16e",x2[0]);
      sprintf(values[4],"%-1.16e",x2[1]);
      sprintf(values[5],"%-1.16e",x2[2]);
      sprintf(values[6],"%-1.16e",x3[0]);
      sprintf(values[7],"%-1.16e",x3[1]);
      sprintf(values[8],"%-1.16e",x3[2]);

      // svalues[0] is not used here
      // used when caller of data_atom_bonus() is via read_data command

      svalues[1] = (const char *) values[0];
      svalues[2] = (const char *) values[1];
      svalues[3] = (const char *) values[2];
      svalues[4] = (const char *) values[3];
      svalues[5] = (const char *) values[4];
      svalues[6] = (const char *) values[5];
      svalues[7] = (const char *) values[6];
      svalues[8] = (const char *) values[7];
      svalues[9] = (const char *) values[8];
      
      avec_tri->data_atom_bonus(n,svalues);

      // allocate all vectors in Connect3d
      // copy neigh_e123 and neigh_c123 from global connect3dall to local connect3d
      //   reset global tri indices to new tri IDs beyond idmaxall
      // other vectors will be set in connectivity3d_complete()

      if (nlocal_connect == nmax_connect) grow_connect();
      memcpy(&connect3d[nlocal_connect],&connect3dall[i],sizeof(Connect3d));

      num = connect3d[nlocal_connect].ne1;
      if (num) {
        connect3d[nlocal_connect].neigh_e1 = tcp->get(num,pool3d[nlocal_connect].neigh_e1);
        connect3d[nlocal_connect].ewhich_e1 = tcp->get(num,pool3d[nlocal_connect].ewhich_e1);
        connect3d[nlocal_connect].nside_e1 = tcp->get(num,pool3d[nlocal_connect].nside_e1);
        connect3d[nlocal_connect].aflag_e1 = tcp->get(num,pool3d[nlocal_connect].aflag_e1);
        global = connect3dall[nlocal_connect].neigh_e1;
        local = connect3d[nlocal_connect].neigh_e1;
        for (j = 0; j < num; j++)
          local[j] = global[j] + idmaxall + 1;
      } else {
        connect3d[nlocal_connect].neigh_e1 = nullptr;
        connect3d[nlocal_connect].ewhich_e1 = nullptr;
        connect3d[nlocal_connect].nside_e1 = nullptr;
        connect3d[nlocal_connect].aflag_e1 = nullptr;
      }

      num = connect3d[nlocal_connect].ne2;
      if (num) {
        connect3d[nlocal_connect].neigh_e2 = tcp->get(num,pool3d[nlocal_connect].neigh_e2);
        connect3d[nlocal_connect].ewhich_e2 = tcp->get(num,pool3d[nlocal_connect].ewhich_e2);
        connect3d[nlocal_connect].nside_e2 = tcp->get(num,pool3d[nlocal_connect].nside_e2);
        connect3d[nlocal_connect].aflag_e2 = tcp->get(num,pool3d[nlocal_connect].aflag_e2);
        global = connect3dall[nlocal_connect].neigh_e2;
        local = connect3d[nlocal_connect].neigh_e2;
        for (j = 0; j < num; j++)
          local[j] = global[j] + idmaxall + 1;
      } else {
        connect3d[nlocal_connect].neigh_e2 = nullptr;
        connect3d[nlocal_connect].ewhich_e2 = nullptr;
        connect3d[nlocal_connect].nside_e2 = nullptr;
        connect3d[nlocal_connect].aflag_e2 = nullptr;
      }

      num = connect3d[nlocal_connect].ne3;
      if (num) {
        connect3d[nlocal_connect].neigh_e3 = tcp->get(num,pool3d[nlocal_connect].neigh_e3);
        connect3d[nlocal_connect].ewhich_e3 = tcp->get(num,pool3d[nlocal_connect].ewhich_e3);
        connect3d[nlocal_connect].nside_e3 = tcp->get(num,pool3d[nlocal_connect].nside_e3);
        connect3d[nlocal_connect].aflag_e3 = tcp->get(num,pool3d[nlocal_connect].aflag_e3);
        global = connect3dall[nlocal_connect].neigh_e3;
        local = connect3d[nlocal_connect].neigh_e3;
        for (j = 0; j < num; j++)
          local[j] = global[j] + idmaxall + 1;
      } else {
        connect3d[nlocal_connect].neigh_e3 = nullptr;
        connect3d[nlocal_connect].ewhich_e3 = nullptr;
        connect3d[nlocal_connect].nside_e3 = nullptr;
        connect3d[nlocal_connect].aflag_e3 = nullptr;
      }

      num = connect3d[nlocal_connect].nc1;
      if (num) {
        connect3d[nlocal_connect].neigh_c1 = tcp->get(num,pool3d[nlocal_connect].neigh_c1);
        connect3d[nlocal_connect].cwhich_c1 = tcp->get(num,pool3d[nlocal_connect].cwhich_c1);
        global = connect3dall[nlocal_connect].neigh_c1;
        local = connect3d[nlocal_connect].neigh_c1;
        for (j = 0; j < num; j++)
          local[j] = global[j] + idmaxall + 1;
      } else {
        connect3d[nlocal_connect].neigh_c1 = nullptr;
        connect3d[nlocal_connect].cwhich_c1 = nullptr;
      }

      num = connect3d[nlocal_connect].nc2;
      if (num) {
        connect3d[nlocal_connect].neigh_c2 = tcp->get(num,pool3d[nlocal_connect].neigh_c2);
        connect3d[nlocal_connect].cwhich_c2 = tcp->get(num,pool3d[nlocal_connect].cwhich_c2);
        global = connect3dall[nlocal_connect].neigh_c2;
        local = connect3d[nlocal_connect].neigh_c2;
        for (j = 0; j < num; j++)
          local[j] = global[j] + idmaxall + 1;
      } else {
        connect3d[nlocal_connect].neigh_c2 = nullptr;
        connect3d[nlocal_connect].cwhich_c2 = nullptr;
      }

      num = connect3d[nlocal_connect].nc3;
      if (num) {
        connect3d[nlocal_connect].neigh_c3 = tcp->get(num,pool3d[nlocal_connect].neigh_c3);
        connect3d[nlocal_connect].cwhich_c3 = tcp->get(num,pool3d[nlocal_connect].cwhich_c3);
        global = connect3dall[nlocal_connect].neigh_c3;
        local = connect3d[nlocal_connect].neigh_c3;
        for (j = 0; j < num; j++)
          local[j] = global[j] + idmaxall + 1;
      } else {
        connect3d[nlocal_connect].neigh_c3 = nullptr;
        connect3d[nlocal_connect].cwhich_c3 = nullptr;
      }

      connect2atom[nlocal_connect] = n;
      atom2connect[n] = nlocal_connect;
      nlocal_connect++;
    }
  }

  memory->sfree(connect3dall);
  memory->destroy(neigh_e1);
  memory->destroy(neigh_e2);
  memory->destroy(neigh_e3);
  memory->destroy(neigh_c1);
  memory->destroy(neigh_c2);
  memory->destroy(neigh_c3);

  // recreate atom map that includes added tris

  if (atom->map_style) {
    atom->map_init();
    atom->map_set();
  }
}

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// methods for final stages of surf connectivity build, for global or local
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------

/* ----------------------------------------------------------------------
   assign values to all other fields in Connect2d for line connections
   only np1,np2 and neigh_p12 were previously assigned
------------------------------------------------------------------------- */

void FixSurfaceLocal::connectivity2d_complete()
{
  // epssq = square of EPSILON fraction of minimum line length
  // for use in same_point()
  
  if (epssq < 0.0) epsilon_calculate();

  // calculate endpts and normals for all owned+ghost line particles

  double ***endpts,**normals;
  memory->create(endpts,nlocal_connect+nghost_connect,2,3,"surface/local:endpts");
  memory->create(normals,nlocal_connect+nghost_connect,3,"surface/local:normals");

  AtomVecLine *avec = (AtomVecLine *) atom->style_match("line");
  AtomVecLine::Bonus *bonus = avec->bonus;
  double **x = atom->x;
  int *line = atom->line;
  int nall = atom->nlocal + atom->nghost;
  
  int m,iconnect;
  double length,theta,dx,dy;
  double p12[3];
  double zunit[3] = {0.0,0.0,1.0};
    
  for (int i = 0; i < nall; i++) {
    if (line[i] < 0) continue;
    m = line[i];
    length = bonus[m].length;
    theta = bonus[m].theta;
    dx = 0.5*length*cos(theta);
    dy = 0.5*length*sin(theta);

    iconnect = atom2connect[i];

    endpts[iconnect][0][0] = x[i][0] - dx;
    endpts[iconnect][0][1] = x[i][1] - dy;
    endpts[iconnect][0][2] = 0.0;
    endpts[iconnect][1][0] = x[i][0] + dx;
    endpts[iconnect][1][1] = x[i][1] + dy;
    endpts[iconnect][1][2] = 0.0;

    MathExtra::sub3(endpts[iconnect][1],endpts[iconnect][0],p12);
    MathExtra::cross3(zunit,p12,normals[iconnect]);
    MathExtra::norm3(normals[iconnect]);
  }

  // set connect2d pwhich/nside/aflag for each end point of each line
  // see fsl.h file for an explanation of each vector in Connect2d
  // aflag is based on dot and cross product of 2 connected line normals
  //   cross product is either along +z or -z direction
  
  int nlocal = atom->nlocal;
 
  int i,j,jconnect;
  tagint jtag;
  double rsq,dotline,dotnorm;
  double *inorm,*jnorm;
  double icrossj[3];

  for (i = 0; i < nlocal; i++) {
    if (line[i] < 0) continue;
    iconnect = atom2connect[i];
    
    for (m = 0; m < connect2d[iconnect].np1; m++) {
      jtag = connect2d[iconnect].neigh_p1[m];
      j = atom->map(jtag);
      jconnect = atom2connect[j];

      inorm = normals[iconnect];
      jnorm = normals[jconnect];
      dotnorm = MathExtra::dot3(inorm,jnorm);

      if (same_point(endpts[iconnect][0],endpts[jconnect][0])) {
        connect2d[iconnect].pwhich_p1[m] = 0;
        connect2d[iconnect].nside_p1[m] = OPPOSITE_SIDE;
        if (dotnorm < -1.0+flatthresh) {
          connect2d[iconnect].aflag_p1[m] = FLAT;
        } else {
          MathExtra::cross3(inorm,jnorm,icrossj);
          if (icrossj[2] > 0.0) connect2d[iconnect].aflag_p1[m] = CONCAVE;
          else connect2d[iconnect].aflag_p1[m] = CONVEX;
        }
      } else if (same_point(endpts[iconnect][0],endpts[jconnect][1])) {
        connect2d[iconnect].pwhich_p1[m] = 1;
        connect2d[iconnect].nside_p1[m] = SAME_SIDE;
        if (dotnorm > 1.0-flatthresh) {
          connect2d[iconnect].aflag_p1[m] = FLAT;
        } else {
          MathExtra::cross3(inorm,jnorm,icrossj);
          if (icrossj[2] < 0.0) connect2d[iconnect].aflag_p1[m] = CONCAVE;
          else connect2d[iconnect].aflag_p1[m] = CONVEX;
        }
      }
    }
    
    for (m = 0; m < connect2d[iconnect].np2; m++) {
      jtag = connect2d[iconnect].neigh_p2[m];
      j = atom->map(jtag);
      jconnect = atom2connect[j];

      inorm = normals[iconnect];
      jnorm = normals[jconnect];
      dotnorm = MathExtra::dot3(inorm,jnorm);

      dx = endpts[iconnect][1][0] - endpts[jconnect][0][0];
      dy = endpts[iconnect][1][1] - endpts[jconnect][0][1];
      rsq = dx*dx + dy*dy;

      if (same_point(endpts[iconnect][1],endpts[jconnect][0])) {
        connect2d[iconnect].pwhich_p2[m] = 0;
        connect2d[iconnect].nside_p2[m] = SAME_SIDE;
        if (dotnorm > 1.0-flatthresh) {
          connect2d[iconnect].aflag_p2[m] = FLAT;
        } else {
          MathExtra::cross3(inorm,jnorm,icrossj);
          if (icrossj[2] > 0.0) connect2d[iconnect].aflag_p2[m] = CONCAVE;
          else connect2d[iconnect].aflag_p2[m] = CONVEX;
        }
      } else if (same_point(endpts[iconnect][1],endpts[jconnect][1])) {
        connect2d[iconnect].pwhich_p2[m] = 1;
        connect2d[iconnect].nside_p2[m] = OPPOSITE_SIDE;
        if (dotnorm < -1.0+flatthresh) {
          connect2d[iconnect].aflag_p2[m] = FLAT;
        } else {
          MathExtra::cross3(inorm,jnorm,icrossj);
          if (icrossj[2] < 0.0) connect2d[iconnect].aflag_p2[m] = CONCAVE;
          else connect2d[iconnect].aflag_p2[m] = CONVEX;
        }
      }
    }
  }

  memory->destroy(endpts);
  memory->destroy(normals);
}

/* ----------------------------------------------------------------------
   assign values to all other fields in Connect3d for tri connections
   only ne1,ne2,ne3 and neigh_e123 were previously assigned
   only nc1,nc2,nc3 and neigh_c123 were previously assigned
------------------------------------------------------------------------- */

void FixSurfaceLocal::connectivity3d_complete()
{
  // epssq = square of EPSILON fraction of minimum tri diameter
  // for use in same_point()

  if (epssq < 0.0) epsilon_calculate();
  
  // calculate corner pts and normals for all owned+ghost tri particles

  double ***cpts,**normals;
  memory->create(cpts,nlocal_connect+nghost_connect,3,3,"surface/local:cpts");
  memory->create(normals,nlocal_connect+nghost_connect,3,"surface/local:normals");

  AtomVecTri *avec = (AtomVecTri *) atom->style_match("tri");
  AtomVecTri::Bonus *bonus = avec->bonus;
  double **x = atom->x;
  double **omega = atom->omega;
  double **angmom = atom->angmom;
  int *tri = atom->tri;
  int nall = atom->nlocal + atom->nghost;
  
  int m,iconnect;
  double p[3][3],p12[3],p13[3];

  for (int i = 0; i < nall; i++) {
    if (tri[i] < 0) continue;
    m = tri[i];
    iconnect = atom2connect[i];

    MathExtra::quat_to_mat(bonus[m].quat,p);
    MathExtra::matvec(p,bonus[m].c1,cpts[iconnect][0]);
    MathExtra::add3(x[i],cpts[iconnect][0],cpts[iconnect][0]);
    MathExtra::matvec(p,bonus[m].c2,cpts[iconnect][1]);
    MathExtra::add3(x[i],cpts[iconnect][1],cpts[iconnect][1]);
    MathExtra::matvec(p,bonus[m].c3,cpts[iconnect][2]);
    MathExtra::add3(x[i],cpts[iconnect][2],cpts[iconnect][2]);

    MathExtra::sub3(cpts[iconnect][1],cpts[iconnect][0],p12);
    MathExtra::sub3(cpts[iconnect][2],cpts[iconnect][0],p13);
    MathExtra::cross3(p12,p13,normals[iconnect]);
    MathExtra::norm3(normals[iconnect]);
  }

  // set connect3d edge ewhich/nside/aflag for each edge of each tri
  // see fsl.h file for an explanation of each edge vector in Connect3d
  // aflag is based on dot and cross product of 2 connected tri normals
  //   cross product is either along itri edge or in opposite dir
  
  int nlocal = atom->nlocal;
 
  int i,j,jconnect,jpfirst,jpsecond;
  tagint jtag;
  double dotline,dotnorm;
  double *inorm,*jnorm;
  double icrossj[3],iedge[3];

  for (i = 0; i < nlocal; i++) {
    if (tri[i] < 0) continue;
    iconnect = atom2connect[i];

    for (m = 0; m < connect3d[iconnect].ne1; m++) {
      jtag = connect3d[iconnect].neigh_e1[m];
      j = atom->map(jtag);
      jconnect = atom2connect[j];

      if (same_point(cpts[iconnect][0],cpts[jconnect][0])) jpfirst = 1;
      else if (same_point(cpts[iconnect][0],cpts[jconnect][1])) jpfirst = 2;
      else if (same_point(cpts[iconnect][0],cpts[jconnect][2])) jpfirst = 3;

      if (same_point(cpts[iconnect][1],cpts[jconnect][0])) jpsecond = 1;
      else if (same_point(cpts[iconnect][1],cpts[jconnect][1])) jpsecond = 2;
      else if (same_point(cpts[iconnect][1],cpts[jconnect][2])) jpsecond = 3;

      inorm = normals[iconnect];
      jnorm = normals[jconnect];
      dotnorm = MathExtra::dot3(inorm,jnorm);
      MathExtra::sub3(cpts[iconnect][1],cpts[iconnect][0],iedge);

      if ((jpfirst == 1 && jpsecond == 2) ||
          (jpfirst == 2 && jpsecond == 3) ||
          (jpfirst == 3 && jpsecond == 1)) {
        connect3d[iconnect].ewhich_e1[m] = jpfirst - 1;
        connect3d[iconnect].nside_e1[m] = OPPOSITE_SIDE;
        if (dotnorm < -1.0+flatthresh) connect3d[iconnect].aflag_e1[m] = FLAT;
        else {
          MathExtra::cross3(inorm,jnorm,icrossj);
          if (MathExtra::dot3(icrossj,iedge) > 0.0)
            connect3d[iconnect].aflag_e1[m] = CONCAVE;
          else
            connect3d[iconnect].aflag_e1[m] = CONVEX;
        }
      } else {
        if (jpfirst == 2) connect3d[iconnect].ewhich_e1[m] = 0;
        else if (jpfirst == 3) connect3d[iconnect].ewhich_e1[m] = 1;
        else if (jpfirst == 1) connect3d[iconnect].ewhich_e1[m] = 2;
        connect3d[iconnect].nside_e1[m] = SAME_SIDE;
        if (dotnorm > 1.0-flatthresh) connect3d[iconnect].aflag_e1[m] = FLAT;
        else {
          MathExtra::cross3(inorm,jnorm,icrossj);
          if (MathExtra::dot3(icrossj,iedge) < 0.0)
            connect3d[iconnect].aflag_e1[m] = CONCAVE;
          else
            connect3d[iconnect].aflag_e1[m] = CONVEX;
        }
      }
    }

    for (m = 0; m < connect3d[iconnect].ne2; m++) {
      jtag = connect3d[iconnect].neigh_e2[m];
      j = atom->map(jtag);
      jconnect = atom2connect[j];

      if (same_point(cpts[iconnect][1],cpts[jconnect][0])) jpfirst = 1;
      else if (same_point(cpts[iconnect][1],cpts[jconnect][1])) jpfirst = 2;
      else if (same_point(cpts[iconnect][1],cpts[jconnect][2])) jpfirst = 3;

      if (same_point(cpts[iconnect][2],cpts[jconnect][0])) jpsecond = 1;
      else if (same_point(cpts[iconnect][2],cpts[jconnect][1])) jpsecond = 2;
      else if (same_point(cpts[iconnect][2],cpts[jconnect][2])) jpsecond = 3;

      inorm = normals[iconnect];
      jnorm = normals[jconnect];
      dotnorm = MathExtra::dot3(inorm,jnorm);
      MathExtra::sub3(cpts[iconnect][2],cpts[iconnect][1],iedge);

      if ((jpfirst == 1 && jpsecond == 2) ||
          (jpfirst == 2 && jpsecond == 3) ||
          (jpfirst == 3 && jpsecond == 1)) {
        connect3d[iconnect].ewhich_e2[m] = jpfirst - 1;
        connect3d[iconnect].nside_e2[m] = OPPOSITE_SIDE;
        if (dotnorm < -1.0+flatthresh) connect3d[iconnect].aflag_e2[m] = FLAT;
        else {
          MathExtra::cross3(inorm,jnorm,icrossj);
          if (MathExtra::dot3(icrossj,iedge) > 0.0)
            connect3d[iconnect].aflag_e2[m] = CONCAVE;
          else
            connect3d[iconnect].aflag_e2[m] = CONVEX;
        }
      } else {
        if (jpfirst == 2) connect3d[iconnect].ewhich_e2[m] = 0;
        else if (jpfirst == 3) connect3d[iconnect].ewhich_e2[m] = 1;
        else if (jpfirst == 1) connect3d[iconnect].ewhich_e2[m] = 2;
        connect3d[iconnect].nside_e2[m] = SAME_SIDE;
        if (dotnorm > 1.0-flatthresh) connect3d[iconnect].aflag_e2[m] = FLAT;
        else {
          MathExtra::cross3(inorm,jnorm,icrossj);
          if (MathExtra::dot3(icrossj,iedge) < 0.0)
            connect3d[iconnect].aflag_e2[m] = CONCAVE;
          else
            connect3d[iconnect].aflag_e2[m] = CONVEX;
        }
      }
    }

    for (m = 0; m < connect3d[iconnect].ne3; m++) {
      jtag = connect3d[iconnect].neigh_e3[m];
      j = atom->map(jtag);
      jconnect = atom2connect[j];

      if (same_point(cpts[iconnect][2],cpts[jconnect][0])) jpfirst = 1;
      else if (same_point(cpts[iconnect][2],cpts[jconnect][1])) jpfirst = 2;
      else if (same_point(cpts[iconnect][2],cpts[jconnect][2])) jpfirst = 3;

      if (same_point(cpts[iconnect][0],cpts[jconnect][0])) jpsecond = 1;
      else if (same_point(cpts[iconnect][0],cpts[jconnect][1])) jpsecond = 2;
      else if (same_point(cpts[iconnect][0],cpts[jconnect][2])) jpsecond = 3;

      inorm = normals[iconnect];
      jnorm = normals[jconnect];
      dotnorm = MathExtra::dot3(inorm,jnorm);
      MathExtra::sub3(cpts[iconnect][0],cpts[iconnect][2],iedge);

      if ((jpfirst == 1 && jpsecond == 2) ||
          (jpfirst == 2 && jpsecond == 3) ||
          (jpfirst == 3 && jpsecond == 1)) {
        connect3d[iconnect].ewhich_e3[m] = jpfirst - 1;
        connect3d[iconnect].nside_e3[m] = OPPOSITE_SIDE;
        if (dotnorm < -1.0+flatthresh) connect3d[iconnect].aflag_e3[m] = FLAT;
        else {
          MathExtra::cross3(inorm,jnorm,icrossj);
          if (MathExtra::dot3(icrossj,iedge) > 0.0)
            connect3d[iconnect].aflag_e3[m] = CONCAVE;
          else
            connect3d[iconnect].aflag_e3[m] = CONVEX;
        }
      } else {
        if (jpfirst == 2) connect3d[iconnect].ewhich_e3[m] = 0;
        else if (jpfirst == 3) connect3d[iconnect].ewhich_e3[m] = 1;
        else if (jpfirst == 1) connect3d[iconnect].ewhich_e3[m] = 2;
        connect3d[iconnect].nside_e3[m] = SAME_SIDE;
        if (dotnorm > 1.0-flatthresh) connect3d[iconnect].aflag_e3[m] = FLAT;
        else {
          MathExtra::cross3(inorm,jnorm,icrossj);
          if (MathExtra::dot3(icrossj,iedge) < 0.0)
            connect3d[iconnect].aflag_e3[m] = CONCAVE;
          else
            connect3d[iconnect].aflag_e3[m] = CONVEX;
        }
      }
    }
  }

  // set connect3d cwhich for each corner point of each tri
  // see fsl.h file for an explanation of each corner vector in Connect3d

  for (int i = 0; i < nlocal; i++) {
    if (tri[i] < 0) continue;
    iconnect = atom2connect[i];

    for (m = 0; m < connect3d[iconnect].nc1; m++) {
      jtag = connect3d[iconnect].neigh_c1[m];
      j = atom->map(jtag);
      jconnect = atom2connect[j];
      if (same_point(cpts[iconnect][0],cpts[jconnect][0])) 
        connect3d[iconnect].cwhich_c1[m] = 0;
      else if (same_point(cpts[iconnect][0],cpts[jconnect][1])) 
        connect3d[iconnect].cwhich_c1[m] = 1;
      else if (same_point(cpts[iconnect][0],cpts[jconnect][2])) 
        connect3d[iconnect].cwhich_c1[m] = 2;
    }
    
    for (m = 0; m < connect3d[iconnect].nc2; m++) {
      jtag = connect3d[iconnect].neigh_c2[m];
      j = atom->map(jtag);
      jconnect = atom2connect[j];
      if (same_point(cpts[iconnect][1],cpts[jconnect][0])) 
        connect3d[iconnect].cwhich_c2[m] = 0;
      else if (same_point(cpts[iconnect][1],cpts[jconnect][1])) 
        connect3d[iconnect].cwhich_c2[m] = 1;
      else if (same_point(cpts[iconnect][1],cpts[jconnect][2])) 
        connect3d[iconnect].cwhich_c2[m] = 2;
    }
    
    for (m = 0; m < connect3d[iconnect].nc3; m++) {
      jtag = connect3d[iconnect].neigh_c3[m];
      j = atom->map(jtag);
      jconnect = atom2connect[j];
      if (same_point(cpts[iconnect][2],cpts[jconnect][0])) 
        connect3d[iconnect].cwhich_c3[m] = 0;
      else if (same_point(cpts[iconnect][2],cpts[jconnect][1])) 
        connect3d[iconnect].cwhich_c3[m] = 1;
      else if (same_point(cpts[iconnect][2],cpts[jconnect][2])) 
        connect3d[iconnect].cwhich_c3[m] = 2;
    }
  }

  // clean up
  
  memory->destroy(cpts);
  memory->destroy(normals);
}

/* ----------------------------------------------------------------------
   return 1 if two points are same within eps
   else return 0
------------------------------------------------------------------------- */

int FixSurfaceLocal::same_point(double *pt1, double *pt2)
{
  double dx = pt1[0] - pt2[0];
  double dy = pt1[1] - pt2[1];
  double dz = pt1[2] - pt2[2];
  double rsq = dx*dx + dy*dy + dz*dz;
  if (rsq < epssq) return 1;
  return 0;
}

/* ----------------------------------------------------------------------
   calculate eps = threshold for matching 2 points from different lines/tris
   stored as eps and epssq
------------------------------------------------------------------------- */

void FixSurfaceLocal::epsilon_calculate()
{
  if (dimension == 2) {
    AtomVecLine::Bonus *bonus = avec_line->bonus;
    int *line = atom->line;
    int nlocal = atom->nlocal;

    double minlen = BIG;
    for (int i = 0; i < nlocal; i++)
      if (line[i] >= 0) minlen = MIN(minlen,bonus[line[i]].length);

    // eps = EPSILON fraction of minimum line length
    
    MPI_Allreduce(&minlen,&eps,1,MPI_DOUBLE,MPI_MIN,world);
    eps *= EPSILON;
    epssq = eps*eps;
    
  } else if (dimension == 3) {
    double *radius = atom->radius;
    int *tri = atom->tri;
    int nlocal = atom->nlocal;
    
    double minlen = BIG;
    for (int i = 0; i < nlocal; i++)
      if (tri[i] >= 0) minlen = MIN(minlen,radius[i]);
    minlen *= 2.0;

    // eps = EPSILON fraction of minimum tri diameter

    MPI_Allreduce(&minlen,&eps,1,MPI_DOUBLE,MPI_MIN,world);
    eps *= EPSILON;
    epssq = eps*eps;
  }
}
