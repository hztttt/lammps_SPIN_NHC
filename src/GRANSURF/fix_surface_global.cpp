// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

// Conceptual decisions
// NOTE: allow for multiple instances of this fix or not ?
// NOTE: warn for too-small lines - but how to know smallest particle size ?
// NOTE: alter connection info if 2 lines/tris are different types ?
// NOTE: should this fix produce any output
//         global array with force on each surf
//         or global array of forces per molecule ID (assuemd consecutive) ?
//         or global vector of per-surf particle contact counts ?
// NOTE: what about reduced vs box units in fix_modify move params like fix_move ?
// NOTE: what about PBC
//       connection finding, for moving surfs, surfs which overlap PBC
//       how is this handled for local surfs
// NOTE: could allow non-assignment of type pairs
//       to enable some particles to pass thru some surfs

// Performance improvements
// NOTE: optimal access to velocity of each surf, depends on motion
// NOTE: need to order connections with FLAT first ?
// NOTE: more efficient neighbor lists, see Joel's 18 Nov email for ideas

// NOTE: Possible to check that motion includes all lines/tris in
//       a connected object?  But not easily possible for local surfs ?

#include "fix_surface_global.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix_neigh_history.h"
#include "force.h"
#include "granular_model.h"
#include "gran_sub_mod.h"
#include "input.h"
#include "lattice.h"
#include "math_const.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "molecule.h"
#include "my_page.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "region.h"
#include "stl_reader.h"
#include "surf_extra.h"
#include "tokenizer.h"
#include "update.h"
#include "variable.h"

#include <algorithm>
#include <map>
#include <tuple>
#include <unordered_set>
#include <vector>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace Granular_NS;
using namespace MathConst;
using namespace MathExtra;
using namespace SurfExtra;

enum{SPHERE,LINE,TRI};           // also in DumpImage
enum{LINEAR,WIGGLE,ROTATE,TRANSROT,VARIABLE};

enum{FLAT,CONCAVE,CONVEX};
enum{SAME_SIDE,OPPOSITE_SIDE};

#define FLATTHRESH 1.0-cos(MY_PI/180.0)    // default = 1 degree
#define DELTA 128
#define DELTACONTACTS 4
#define DELTAMODEL 1    // make larger after debugging
#define DELTAMOTION 1   // make larger after debugging
#define MAXSURFTYPE 1024  // extreme, so can reduce it later
#define BIG 1.0e20

/* ---------------------------------------------------------------------- */

FixSurfaceGlobal::FixSurfaceGlobal(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), tstr(nullptr)
{
  if (!atom->radius_flag || !atom->omega_flag)
    error->all(FLERR,"Fix surface/global requires atom attributes radius and omega");

  dimension = domain->dimension;

  // process one or more inputs
  // read triangles/lines from molecule template IDs or STL files
  // hash = map to store unique points
  //   key = xyz coords of a point
  //   value = index into unique points vector

  npoints = maxpoints = 0;
  nlines = ntris = 0;
  points = nullptr;
  lines = nullptr;
  tris = nullptr;

  int ninput = 0;
  std::map<std::tuple<double,double,double>,int> *hash =
    new std::map<std::tuple<double,double,double>,int>();

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"input") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix surface/global command");
      if (strcmp(arg[iarg+1],"mol") == 0) {
        if (iarg+3 > narg) error->all(FLERR,"Illegal fix surface/global command");
        extract_from_molecule(arg[iarg+2],hash);
        iarg += 3;
      } else if (strcmp(arg[iarg+1],"stl") == 0) {
        if (iarg+4 > narg) error->all(FLERR,"Illegal fix surface/global command");
        int stype = utils::inumeric(FLERR,arg[iarg+2],false,lmp);
        extract_from_stlfile(arg[iarg+3],stype,hash);
        iarg += 4;
      } else error->all(FLERR,"Illegal fix surface/global command");
    } else break;

    ninput++;
  }

  delete hash;
  if (ninput == 0)
    error->all(FLERR,"Fix surface/global command requires input keyword");

  // process one or more granular models
  // disable bonded/history option for now

  models = nullptr;
  nmodel = maxmodel = 0;
  heat_flag = 0;
  use_history = 0;
  size_history = -1;

  while (iarg < narg) {
    if (strcmp(arg[iarg],"model") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix surface/global command");

      if (nmodel == maxmodel) {
        maxmodel += DELTAMODEL;
        modeltypes = (ModelTypes *)
          memory->srealloc(models,maxmodel*sizeof(ModelTypes),"surf/global:modeltypes");
        models = (Granular_NS::GranularModel **)
          memory->srealloc(models,maxmodel*sizeof(Granular_NS::GranularModel *),
                           "surf/global:models");
      }

      models[nmodel] = model = new GranularModel(lmp);

      // assign range of particle and surf types to this model
      // ues MAXSURFTYPE for now, in case smax keyword extends input surf types

      utils::bounds(FLERR, arg[iarg+1], 1, atom->ntypes,
                    modeltypes[nmodel].plo, modeltypes[nmodel].phi, error);
      utils::bounds(FLERR, arg[iarg+2], 1, MAXSURFTYPE,
                    modeltypes[nmodel].slo, modeltypes[nmodel].shi, error);
      nmodel++;

      model->contact_type = SURFACE;

      int classic_flag = 1;
      iarg += 3;
      if (strcmp(arg[iarg], "granular") == 0) {
        classic_flag = 0;
        iarg += 1;
      }

      if (classic_flag) {
        iarg = model->define_classic_model(arg, iarg, narg);
        if (iarg < narg && strcmp(arg[iarg], "limit_damping") == 0) {
          model->limit_damping = 1;
          iarg++;
        }
      } else {
        iarg = model->add_sub_model(arg, iarg, narg, NORMAL);
        while (iarg < narg) {
          if (strcmp(arg[iarg], "damping") == 0) {
            iarg = model->add_sub_model(arg, iarg + 1, narg, DAMPING);
          } else if (strcmp(arg[iarg], "tangential") == 0) {
            iarg = model->add_sub_model(arg, iarg + 1, narg, TANGENTIAL);
          } else if (strcmp(arg[iarg], "rolling") == 0) {
            iarg = model->add_sub_model(arg, iarg + 1, narg, ROLLING);
          } else if (strcmp(arg[iarg], "twisting") == 0) {
            iarg = model->add_sub_model(arg, iarg + 1, narg, TWISTING);
          } else if (strcmp(arg[iarg], "heat") == 0) {
            iarg = model->add_sub_model(arg, iarg + 1, narg, HEAT);
            heat_flag = 1;
          } else if (strcmp(arg[iarg], "limit_damping") == 0) {
            model->limit_damping = 1;
            iarg++;
          } else break;
        }
      }

      // define default damping sub model
      // if unspecified, takes no args
      // JOEL NOTE: is damping_model check only for granular or also classic ?
      //    ANSWER: it's performed for both, but the classic model always has damping

      if (!model->damping_model) model->construct_sub_model("viscoelastic", DAMPING);

      model->init();

      // JOEL NOTE: do size_history and use_history apply to each model or all models?
      //    ANSWER: all models, logic updated below

      if (model->beyond_contact) size_history = MAX(size_history + 1, model->size_history);
      else size_history = MAX(size_history, model->size_history);
      if (model->size_history != 0) use_history = 1;
    } else break;
  }

  if (nmodel == 0)
    error->all(FLERR,"Fix surface/global command requires model keyword");

  // maxsurftype = max surf type of any input surf (for now)

  maxsurftype = 0;
  if (dimension == 2) {
    for (int i = 0; i < nlines; i++)
      maxsurftype = MAX(maxsurftype,lines[i].type);
  } else {
    for (int i = 0; i < ntris; i++)
      maxsurftype = MAX(maxsurftype,tris[i].type);
  }

  // optional command-line args
  // smaxtype overrides max surf type of input surfs
  // flat overrides FLATTHRESH of one degree

  int Twall_defined = 0;
  flatthresh = FLATTHRESH;

  while (iarg < narg) {
    if (strcmp(arg[iarg],"smaxtype") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix surface/global command");
      int smaxtype = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      if (smaxtype > MAXSURFTYPE)
        error->all(FLERR,"Fix surface/global smaxtype > MAXSURFTYPE");
      if (smaxtype < maxsurftype)
        error->all(FLERR,"Fix surface/global smaxtype < input surf types");
      maxsurftype = smaxtype;
      iarg += 2;
    } else if (strcmp(arg[iarg],"flat") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix surface/global command");
      double flat = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (flat < 0.0 || flat > 90.0)
        error->all(FLERR,"Invalid value for fix surface/global flat");
      flatthresh = 1.0 - cos(MY_PI*flat/180.0);
      iarg += 2;
    } else if (strcmp(arg[iarg],"temperature") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix surface/global command");
      if (utils::strmatch(arg[iarg+1], "^v_")) {
        tstr = utils::strdup(arg[iarg+1] + 2);
      } else {
        Twall = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      }
      Twall_defined = 1;
      iarg += 2;
    } else error->all(FLERR,"Illegal fix surface/global command");
  }

  if (heat_flag && !Twall_defined)
    error->all(FLERR, "Must define wall temperature with a heat model");

  // reset modeltypes shi from MAXSURFTYPE to maxsurtype
  // initialize types2model for all particle/surf type pairs
  // check that a model has been assigned to every type pair

  for (int i = 0; i < nmodel; i++)
    if (modeltypes[i].shi == MAXSURFTYPE) modeltypes[i].shi = maxsurftype;

  types2model = new Granular_NS::GranularModel**[atom->ntypes+1];
  for (int i = 1; i <= atom->ntypes; i++)
    types2model[i] = new Granular_NS::GranularModel*[maxsurftype+1];
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = 1; j <= maxsurftype; j++)
      types2model[i][j] = nullptr;

  for (int m = 0; m < nmodel; m++)
    for (int i = modeltypes[m].plo; i <= modeltypes[m].phi; i++)
      for (int j = modeltypes[m].slo; j <= modeltypes[m].shi; j++)
        types2model[i][j] = models[m];

  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = 1; j <= maxsurftype; j++)
      if (!types2model[i][j])
        error->all(FLERR,"Fix surface/global type pair is missing a granular model");

  // initializations

  if (dimension == 2) nsurf = nlines;
  else nsurf = ntris;

  nmotion = maxmotion = 0;
  motions = NULL;
  anymove = anymove_variable = 0;

  points_lastneigh = nullptr;
  points_original = nullptr;
  xsurf_original = nullptr;
  pointmove = nullptr;

  neigh_p1 = neigh_p2 = nullptr;
  pwhich_p1 = pwhich_p2 = nullptr;
  nside_p1 = nside_p2 = nullptr;
  aflag_p1 = aflag_p2 = nullptr;

  neigh_e1 = neigh_e2 = neigh_e3 = nullptr;
  ewhich_e1 = ewhich_e2 = ewhich_e3 = nullptr;
  nside_e1 = nside_e2 = nside_e3 =nullptr;
  aflag_e1 = aflag_e2 = aflag_e3 = nullptr;
  neigh_c1 = neigh_c2 = neigh_c3 = nullptr;
  cwhich_c1 = cwhich_c2 = cwhich_c3 = nullptr;

  connect2d = nullptr;
  connect3d = nullptr;

  xsurf = vsurf = omegasurf = nullptr;
  radsurf = nullptr;

  contact_surfs = nullptr;
  nmax_contact_surfs = 0;

  nmax = 0;
  mass_rigid = nullptr;

  fix_rigid = nullptr;
  fix_history = nullptr;

  list = new NeighList(lmp);
  if (use_history) {
    listhistory = new NeighList(lmp);
    zeroes = new double[size_history];
    for (int i = 0; i < size_history; i++) zeroes[i] = 0.0;
  } else {
    listhistory = nullptr;
    zeroes = nullptr;
  }

  imax = 0;
  imflag = nullptr;
  imdata = nullptr;

  type2motion = new int[maxsurftype+1];

  firsttime = 1;

  // initialize surface attributes

  surface_attributes();

  // error checks on duplicate surfs or zero-size surfs

  if (dimension == 2) check2d();
  else check3d();

  // compute connectivity of triangles/lines
  // create Connect3d or Connect2d data structs

  if (dimension == 2) connectivity2d();
  else connectivity3d();

  // warn if any connections between surfs with different molIDs

  check_molecules();

  // print stats on surfs and their connectivity

  if (dimension == 2) stats2d();
  else stats3d();
}

/* ---------------------------------------------------------------------- */

FixSurfaceGlobal::~FixSurfaceGlobal()
{
  memory->sfree(points);
  memory->sfree(lines);
  memory->sfree(tris);

  memory->destroy(points_lastneigh);
  memory->destroy(points_original);
  memory->destroy(xsurf_original);
  memory->destroy(pointmove);

  memory->sfree(modeltypes);
  for (int i = 0; i < nmodel; i++) delete models[i];
  memory->sfree(models);

  for (int i = 1; i <= atom->ntypes; i++) delete [] types2model[i];
  delete [] types2model;

  memory->destroy(neigh_p1);
  memory->destroy(neigh_p2);
  memory->destroy(pwhich_p1);
  memory->destroy(pwhich_p2);
  memory->destroy(nside_p1);
  memory->destroy(nside_p2);
  memory->destroy(aflag_p1);
  memory->destroy(aflag_p2);

  memory->destroy(neigh_e1);
  memory->destroy(neigh_e2);
  memory->destroy(neigh_e3);
  memory->destroy(ewhich_e1);
  memory->destroy(ewhich_e2);
  memory->destroy(ewhich_e3);
  memory->destroy(nside_e1);
  memory->destroy(nside_e2);
  memory->destroy(nside_e3);
  memory->destroy(aflag_e1);
  memory->destroy(aflag_e2);
  memory->destroy(aflag_e3);

  memory->destroy(neigh_c1);
  memory->destroy(neigh_c2);
  memory->destroy(neigh_c3);
  memory->destroy(cwhich_c1);
  memory->destroy(cwhich_c2);
  memory->destroy(cwhich_c3);

  memory->sfree(connect2d);
  memory->sfree(connect3d);

  memory->sfree(contact_surfs);

  memory->destroy(xsurf);
  memory->destroy(vsurf);
  memory->destroy(omegasurf);
  memory->destroy(radsurf);

  memory->destroy(mass_rigid);

  for (int i = 0; i < nmotion; i++) {
    if (motions[i].mstyle == VARIABLE) {
      delete [] motions[i].xvarstr;
      delete [] motions[i].yvarstr;
      delete [] motions[i].zvarstr;
      delete [] motions[i].vxvarstr;
      delete [] motions[i].vyvarstr;
      delete [] motions[i].vzvarstr;
    }
  }

  memory->sfree(motions);
  delete [] type2motion;

  delete list;
  delete listhistory;
  delete [] zeroes;
  delete [] tstr;

  if (use_history)
    modify->delete_fix("NEIGH_HISTORY_SURFACE_GLOBAL_" + std::to_string(instance_me));

  memory->destroy(imflag);
  memory->destroy(imdata);
}

/* ----------------------------------------------------------------------
   create Fix needed for storing shear history if needed
   must be done in post_constructor()
------------------------------------------------------------------------- */

void FixSurfaceGlobal::post_constructor()
{
  if (use_history) {
    auto cmd = fmt::format("NEIGH_HISTORY_SURFACE_GLOBAL_" + std::to_string(instance_me) + " all NEIGH_HISTORY {}",size_history);
    fix_history = dynamic_cast<FixNeighHistory *>(modify->add_fix(cmd));
  } else
    fix_history = nullptr;
}

/* ----------------------------------------------------------------------
   mask for INITIAL_INTEGRATE will be set by fix_modify move
---------------------------------------------------------------------- */

int FixSurfaceGlobal::setmask()
{
  int mask = 0;
  mask |= PRE_NEIGHBOR;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixSurfaceGlobal::init()
{
  dt = update->dt;
  triggersq = 0.25 * neighbor->skin * neighbor->skin;

  // check for compatible heat conduction atom style

  if (heat_flag) {
    if (!atom->temperature_flag)
      error->all(FLERR, "Heat conduction in fix surface/global requires atom style with temperature property");
    if (!atom->heatflow_flag)
      error->all(FLERR, "Heat conduction in fix surface/global requires atom style with heatflow property");
  }

  // define history indices
  // JOEL NOTE: WHat are "beyond" contact models ?
  //            Why is this check not made in constructor ?

  int next_index = 0;
  if (model->beyond_contact) //next_index = 1;
    error->all(FLERR, "Beyond contact models not currenty supported");

  for (int i = 0; i < NSUBMODELS; i++) {
    model->sub_models[i]->history_index = next_index;
    next_index += model->sub_models[i]->size_history;
  }

  // one-time setup and allocation of neighbor list
  // wait until now, so neighbor settings have been made

  if (firsttime) {
    firsttime = 0;
    int pgsize = neighbor->pgsize;
    int oneatom = neighbor->oneatom;
    list->setup_pages(pgsize,oneatom);
    list->grow(atom->nmax,atom->nmax);

    if (use_history) {
      listhistory->setup_pages(pgsize,oneatom);
      listhistory->grow(atom->nmax,atom->nmax);
    }
  }

  if (tstr) {
    tvar = input->variable->find(tstr);
    if (tvar < 0)
      error->all(FLERR, "Variable {} for fix surface/global does not exist", tstr);
    if (! input->variable->equalstyle(tvar))
      error->all(FLERR,
                 "Variable {} for fix surface/global must be an equal style variable",
                 tstr);
  }

  // check on motion variables

  for (int i = 0; i < nmotion; i++) {
    Motion *motion = &motions[i];
    if (motion->mstyle != VARIABLE) continue;

    if (motion->xvarstr) {
      motion->xvar = input->variable->find(motion->xvarstr);
      if (motion->xvar < 0)
        error->all(FLERR, "Variable name for fix_modify move does not exist");
      if (!input->variable->equalstyle(motion->xvar))
        error->all(FLERR, "Variable for fix_modify move must be equal style");
    }
    if (motion->yvarstr) {
      motion->yvar = input->variable->find(motion->yvarstr);
      if (motion->yvar < 0)
        error->all(FLERR, "Variable name for fix_modify move does not exist");
      if (!input->variable->equalstyle(motion->yvar))
        error->all(FLERR, "Variable for fix_modify move must be equal style");
    }
    if (motion->zvarstr) {
      motion->zvar = input->variable->find(motion->zvarstr);
      if (motion->zvar < 0)
        error->all(FLERR, "Variable name for fix_modify move does not exist");
      if (!input->variable->equalstyle(motion->zvar))
        error->all(FLERR, "Variable for fix_modify move must be equal style");
    }

    if (motion->vxvarstr) {
      motion->vxvar = input->variable->find(motion->vxvarstr);
      if (motion->vxvar < 0)
        error->all(FLERR, "Variable name for fix_modify move does not exist");
      if (!input->variable->equalstyle(motion->vxvar))
        error->all(FLERR, "Variable for fix_modify move must be equal style");
    }
    if (motion->vyvarstr) {
      motion->vyvar = input->variable->find(motion->vyvarstr);
      if (motion->vyvar < 0)
        error->all(FLERR, "Variable name for fix_modify move does not exist");
      if (!input->variable->equalstyle(motion->vyvar))
        error->all(FLERR, "Variable for fix_modify move must be equal style");
    }
    if (motion->vzvarstr) {
      motion->vzvar = input->variable->find(motion->vzvarstr);
      if (motion->vzvar < 0)
        error->all(FLERR, "Variable name for fix_modify move does not exist");
      if (!input->variable->equalstyle(motion->vzvar))
        error->all(FLERR, "Variable for fix_modify move must be equal style");
    }
  }

  // initialize pointmove settings
  // fix_modify move can be set between runs

  if (anymove) {
    for (int i = 0; i < npoints; i++) pointmove[i] = 0;
  }
}

/* ---------------------------------------------------------------------- */

void FixSurfaceGlobal::setup_pre_neighbor()
{
  pre_neighbor();
}

/* ----------------------------------------------------------------------
   move surfaces via fix_modify setting
   similar to fix move operations
------------------------------------------------------------------------- */

void FixSurfaceGlobal::initial_integrate(int vflag)
{
  int imotion,mstyle;

  // invoke variables for any VARIABLE style motion

  if (anymove_variable) {
    for (int i = 0; i < nmotion; i++) {
      Motion *motion = &motions[i];
      if (motion->mstyle != VARIABLE) continue;
      if (motion->xvarstr) motion->dx = input->variable->compute_equal(motion->xvar);
      if (motion->yvarstr) motion->dy = input->variable->compute_equal(motion->yvar);
      if (motion->zvarstr) motion->dz = input->variable->compute_equal(motion->zvar);
      if (motion->vxvarstr) motion->vx = input->variable->compute_equal(motion->vxvar);
      if (motion->vyvarstr) motion->vy = input->variable->compute_equal(motion->vyvar);
      if (motion->vzvarstr) motion->vz = input->variable->compute_equal(motion->vzvar);
    }
  }

  // invoke appropriate move option for each surf

  for (int i = 0; i < nsurf; i++) {
    if (dimension == 2) imotion = type2motion[lines[i].type];
    else imotion = type2motion[tris[i].type];
    if (imotion < 0) continue;

    mstyle = motions[imotion].mstyle;
    if (mstyle == LINEAR) move_linear(imotion,i);
    else if (mstyle == WIGGLE) move_wiggle(imotion,i);
    else if (mstyle == ROTATE) move_rotate(imotion,i);
    else if (mstyle == TRANSROT) move_transrotate(imotion,i);
    else if (mstyle == VARIABLE) move_variable(imotion,i);
  }

  // clear pointmove settings

  if (dimension == 2) {
    for (int i = 0; i < nlines; i++) {
      if (type2motion[lines[i].type] < 0) continue;
      pointmove[lines[i].p1] = 0;
      pointmove[lines[i].p2] = 0;
    }
  } else {
    for (int i = 0; i < ntris; i++) {
      if (type2motion[tris[i].type] < 0) continue;
      pointmove[tris[i].p1] = 0;
      pointmove[tris[i].p2] = 0;
      pointmove[tris[i].p3] = 0;
    }
  }

  // trigger reneighbor if any point has moved skin/2 distance

  double dx,dy,dz,rsq;
  double *pt;

  int triggerflag = 0;

  for (int i = 0; i < npoints; i++) {
    pt = points[i].x;
    dx = pt[0] - points_lastneigh[i][0];
    dy = pt[1] - points_lastneigh[i][1];
    dz = pt[2] - points_lastneigh[i][2];
    rsq = dx*dx + dy*dy + dz*dz;
    if (rsq > triggersq) {
      triggerflag = 1;
      break;
    }
  }

  if (triggerflag) next_reneighbor = update->ntimestep;
}

/* ----------------------------------------------------------------------
   build neighbor list for sphere/surf interactions
   I = sphere, J = surf
   similar to methods in neigh_gran.cpp
------------------------------------------------------------------------- */

void FixSurfaceGlobal::pre_neighbor()
{
  int i,j,m,n,nn,dnum,dnumbytes;
  double xtmp,ytmp,ztmp,delx,dely,delz;
  double radi,rsq,radsum,cutsq;
  int *neighptr,*touchptr;
  double *shearptr;

  int *npartner;
  tagint **partner;
  double **shearpartner;
  int **firsttouch;
  double **firstshear;
  MyPage<int> *ipage_touch;
  MyPage<double> *dpage_shear;

  double **x = atom->x;
  double *radius = atom->radius;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  double skin = neighbor->skin;

  list->grow(nlocal,nall);
  if (use_history) listhistory->grow(nlocal,nall);

  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  MyPage<int> *ipage = list->ipage;

  if (use_history) {
    fix_history->nlocal_neigh = nlocal;
    //npartner = fix_history->npartner;
    //partner = fix_history->partner;
    //shearpartner = fix_history->shearpartner;
    firsttouch = fix_history->firstflag;
    firstshear = fix_history->firstvalue;
    //ipage_touch = listhistory->ipage;
    //dpage_shear = listhistory->dpage;
    //dnum = listhistory->dnum;
    dnumbytes = dnum * sizeof(double);
  }

  // store current point positions for future neighbor trigger check
  // check is performed in intitial_integrate()

  if (anymove) {
    for (i = 0; i < npoints; i++) {
      points_lastneigh[i][0] = points[i].x[0];
      points_lastneigh[i][1] = points[i].x[1];
      points_lastneigh[i][2] = points[i].x[2];
    }
  }

  int inum = 0;
  ipage->reset();
  if (use_history) {
    ipage_touch->reset();
    dpage_shear->reset();
  }

  for (i = 0; i < nlocal; i++) {
    n = 0;
    neighptr = ipage->vget();
    if (use_history) {
      nn = 0;
      touchptr = ipage_touch->vget();
      shearptr = dpage_shear->vget();
    }

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    radi = radius[i];

    // for now, loop over all surfs

    for (j = 0; j < nsurf; j++) {
      delx = xtmp - xsurf[j][0];
      dely = ytmp - xsurf[j][1];
      delz = ztmp - xsurf[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      radsum = radi + radsurf[j] + skin;
      cutsq = radsum*radsum;
      if (rsq <= cutsq) {
        neighptr[n] = j;

        if (use_history) {
          if (rsq < radsum*radsum) {
            for (m = 0; m < npartner[i]; m++)
              if (partner[i][m] == j) break;
            if (m < npartner[i]) {
              touchptr[n] = 1;
              memcpy(&shearptr[nn],&shearpartner[i][dnum*m],dnumbytes);
              nn += dnum;
            } else {
              touchptr[n] = 0;
              memcpy(&shearptr[nn],zeroes,dnumbytes);
              nn += dnum;
            }
          } else {
            touchptr[n] = 0;
            memcpy(&shearptr[nn],zeroes,dnumbytes);
            nn += dnum;
          }
        }

        n++;
      }
    }

    ilist[inum++] = i;
    firstneigh[i] = neighptr;
    numneigh[i] = n;
    ipage->vgot(n);
    if (ipage->status())
      error->one(FLERR,"Fix surface/global neighbor list overflow, "
                 "boost neigh_modify one");

    if (use_history) {
      firsttouch[i] = touchptr;
      firstshear[i] = shearptr;
      ipage_touch->vgot(n);
      dpage_shear->vgot(nn);
    }
  }

  list->inum = inum;
}

/* ----------------------------------------------------------------------
   compute particle/surface interactions
   impart force and torque to spherical particles
------------------------------------------------------------------------- */

void FixSurfaceGlobal::post_force(int vflag)
{
  int i,j,k,a,n,m,nconnect,ii,jj,inum,jnum,jflag,otherflag;
  int n_contact_surfs;
  double xtmp,ytmp,ztmp,radi,delx,dely,delz;
  double meff;
  int *ilist,*jlist,*numneigh,**firstneigh;
  int *touch,**firsttouch,touch_flag;
  double rsq,radsum;
  double dr[3],contact[3],ds[3],xc[3],vc[3],omegac[3],*forces,*torquesi;
  double *history,*allhistory,**firsthistory;

  double dot, overlap, max_overlap, tmp[3];
  int it, jjtmp, aflag, pwhich, shared_pt_j, shared_pt_k, convex_flag, nsidej, nsidek, skip;
  std::vector<int> *flat_surfs = new std::vector<int>();
  std::unordered_set<int> *processed_contacts = new std::unordered_set<int>();
  std::unordered_set<int> *hidden_contacts = new std::unordered_set<int>();
  std::map<int, int> *contacts_map = new std::map<int, int>();

  model->history_update = 1;
  model->radj = 0.0;
  if (update->setupflag) model->history_update = 0;

  // if just reneighbored:
  // update rigid body masses for owned atoms if using FixRigid
  //   body[i] = which body atom I is in, -1 if none
  //   mass_body = mass of each rigid body

  if (neighbor->ago == 0 && fix_rigid) {
    int tmp;
    int *body = (int *) fix_rigid->extract("body",tmp);
    double *mass_body = (double *) fix_rigid->extract("masstotal",tmp);
    if (atom->nmax > nmax) {
      memory->destroy(mass_rigid);
      nmax = atom->nmax;
      memory->create(mass_rigid,nmax,"surface/global:mass_rigid");
    }
    int nlocal = atom->nlocal;
    for (i = 0; i < nlocal; i++) {
      if (body[i] >= 0) mass_rigid[i] = mass_body[body[i]];
      else mass_rigid[i] = 0.0;
    }
  }

  // loop over neighbors of my atoms
  // I is always sphere, J is always line

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double **omega = atom->omega;
  double **torque = atom->torque;
  double *radius = atom->radius;
  double *temperature = atom->temperature;
  double *heatflow = atom->heatflow;
  double *rmass = atom->rmass;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  if (heat_flag) {
    if (tstr)
      Twall = input->variable->compute_equal(tvar);
    model->Tj = Twall;
  }

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  if (use_history) {
    firsttouch = fix_history->firstflag;
    firsthistory = fix_history->firstvalue;
  }

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if (!(mask[i] & groupbit)) continue;
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    radi = radius[i];
    model->xi = x[i];
    model->radi = radi;
    model->vi = v[i];
    model->omegai = omega[i];
    if (heat_flag) model->Ti = temperature[i];

    // if I is part of rigid body, use body mass
    meff = rmass[i];
    if (fix_rigid && mass_rigid[i] > 0.0) meff = mass_rigid[i];
    model->meff = meff;

    jlist = firstneigh[i];
    jnum = numneigh[i];
    if (use_history) {
      touch = firsttouch[i];
      allhistory = firsthistory[i];
    }

    n_contact_surfs = 0;
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];

      // Do we need special bonds?
      // factor_lj = special_lj[sbmask(j)];
      // j &= NEIGHMASK;
      // if (factor_lj == 0) continue;

      delx = xtmp - xsurf[j][0];
      dely = ytmp - xsurf[j][1];
      delz = ztmp - xsurf[j][2];
      rsq = delx * delx + dely * dely + delz * delz;

      // skip contact check if particle/surf are too far apart

      radsum = radi + radsurf[j];
      if (rsq > radsum * radsum) {
        if (use_history) {
          touch[jj] = 0;
          history = &allhistory[size_history * jj];
          for (k = 0; k < size_history; k++) history[k] = 0.0;
        }
        continue;
      }

      // check for contact between particle and line/tri

      if (dimension == 2) {

        // check for overlap of sphere and line segment
        // jflag = 0 for no overlap, 1 for interior line pt, -1/-2 for end pts
        // if no overlap, just continue
        // for overlap, also return:
        //   contact = nearest point on line to sphere center
        //   dr = vector from contact pt to sphere center
        //   rsq = squared length of dr

        jflag = SurfExtra::
          overlap_sphere_line(x[i],radius[i],
                              points[lines[j].p1].x,points[lines[j].p2].x,
                              contact,dr,rsq);
      } else {

        // check for overlap of sphere and triangle
        // jflag = 0 for no overlap, 1 for interior line pt,
        //   -1/-2/-3 for 3 edges, -4/-5/-6 for 3 corner pts
        // if no overlap, just continue
        // for overlap, also returns:
        //   contact = nearest point on tri to sphere center
        //   dr = vector from contact pt to sphere center
        //   rsq = squared length of dr

        jflag = SurfExtra::
          overlap_sphere_tri(x[i],radius[i],
                             points[tris[j].p1].x,points[tris[j].p2].x,
                             points[tris[j].p3].x,tris[j].norm,
                             contact,dr,rsq);
      }

      if (!jflag) {
        if (use_history) {
          touch[jj] = 0;
          history = &allhistory[size_history * jj];
          for (k = 0; k < size_history; k++) history[k] = 0.0;
        }
        continue;
      }

      // append surf to list of contacts

      if (n_contact_surfs + 1 > nmax_contact_surfs) {
        nmax_contact_surfs += DELTACONTACTS;
        memory->grow(contact_surfs, nmax_contact_surfs * sizeof(ContactSurf),
                                      "surface/global:contact_surfs");
      }

      if (dimension == 2) {

        // Store which side is in contact relative to normal vector
        dot = MathExtra::dot3(lines[j].norm, dr);
        if (dot >= 0) nsidej = SAME_SIDE;
        else nsidej = OPPOSITE_SIDE;

        contact_surfs[n_contact_surfs].index = j;
        contact_surfs[n_contact_surfs].neigh_index = jj;
        contact_surfs[n_contact_surfs].type = lines[j].type;
        contact_surfs[n_contact_surfs].jflag = jflag;
        contact_surfs[n_contact_surfs].nside = nsidej;
        contact_surfs[n_contact_surfs].overlap = radi - sqrt(rsq);
        contact_surfs[n_contact_surfs].r[0] = dr[0];
        contact_surfs[n_contact_surfs].r[1] = dr[1];
        contact_surfs[n_contact_surfs].r[2] = dr[2];
        contact_surfs[n_contact_surfs].ignore_pt = 0;
      }

      n_contact_surfs += 1;
    }

    // reduce set of contacts

    std::sort(contact_surfs, contact_surfs + n_contact_surfs, [](ContactSurf a, ContactSurf b) {return a.overlap > b.overlap;});

    contacts_map->clear();
    for (n = 0; n < n_contact_surfs; n++)
      (*contacts_map)[contact_surfs[n].index] = n;

    processed_contacts->clear();
    hidden_contacts->clear();

    for (n = 0; n < n_contact_surfs; n++) {
      j = contact_surfs[n].index;

      if (processed_contacts->find(j) != processed_contacts->end()) continue;
      processed_contacts->insert(j);
      flat_surfs->clear();
      flat_surfs->push_back(n);
      nsidej = contact_surfs[n].nside;

      // Loop through connected surfs
      if (dimension == 2) {
        for (nconnect = 0; nconnect < (connect2d[j].np1 + connect2d[j].np2); nconnect++) {
          shared_pt_j = 0;
          if (nconnect < connect2d[j].np1) {
            k = connect2d[j].neigh_p1[nconnect];
            aflag = connect2d[j].aflag_p1[nconnect];
            nsidek = connect2d[j].nside_p1[nconnect];
            pwhich = connect2d[j].pwhich_p1[nconnect];
            if (contact_surfs[n].jflag == -1)
              shared_pt_j = 1;
          } else {
            k = connect2d[j].neigh_p2[nconnect - connect2d[j].np1];
            aflag = connect2d[j].aflag_p2[nconnect - connect2d[j].np1];
            nsidek = connect2d[j].nside_p2[nconnect - connect2d[j].np1];
            pwhich = connect2d[j].pwhich_p2[nconnect - connect2d[j].np1];
            if (contact_surfs[n].jflag == -2)
              shared_pt_j = 1;
          }

          // Skip if not in contact
          if (contacts_map->find(k) == contacts_map->end())
            continue;

          // Skip if already processed
          if (processed_contacts->find(k) != processed_contacts->end())
            continue;

          m = (*contacts_map)[k];
          shared_pt_k = 0;
          if ((pwhich == 0 && contact_surfs[m].jflag == -1) ||
              (pwhich == 1 && contact_surfs[m].jflag == -2))
            shared_pt_k = 1;

          // TODO: how to handle contact_surfs[n].type != lines[k].type?
          if (aflag == FLAT) {
            // flat: recursively walk & process all other flat surfs
            //   store which side is associated with the side of the closest
            //   connected flat surf is in contact with the sphere
            if (nsidej == OPPOSITE_SIDE) {
              if (nsidek == OPPOSITE_SIDE)
                nsidek = SAME_SIDE;
              else
                nsidek = OPPOSITE_SIDE;
            }
            contact_surfs[m].nside = nsidek;

            // check whether to skip end point
            if (shared_pt_j) {
              contact_surfs[n].ignore_pt = 1;
            } if (shared_pt_k) {
              contact_surfs[m].ignore_pt = 1;
            }

            walk_flat_connections2d(k, nsidek, flat_surfs, processed_contacts, hidden_contacts, contacts_map);
          } else {
            // non-flat: skip contribution to force if convex
            convex_flag = 0;
            if ((nsidej == SAME_SIDE && aflag == CONVEX) ||
                (nsidej == OPPOSITE_SIDE && aflag == CONCAVE))
              convex_flag = 1;

            if (convex_flag) {
              hidden_contacts->insert(k);
            } else {
              if (shared_pt_j) {
                contact_surfs[n].ignore_pt = 1;
              } if (shared_pt_k) {
                contact_surfs[m].ignore_pt = 1;
              }
            }
          }
        }
      } else {
        continue; // 2d to start
      }

      // Calculate geometry

      max_overlap = contact_surfs[n].overlap;
      skip = 0;

      if (flat_surfs->size() != 1) {
        // For flat surfs, calculate overlap-weighted average normal vector
        MathExtra::zero3(xc);

        for (it = 0; it < flat_surfs->size(); it++) {
          m = (*flat_surfs)[it];
          overlap = contact_surfs[m].overlap;
          nsidek = contact_surfs[m].nside;
          k = contact_surfs[m].index;
          if (hidden_contacts->find(k) != hidden_contacts->end()) {
            skip = 1;
            break;
          }

          if (contact_surfs[m].ignore_pt) {
            // extend surface to get normal vector
            MathExtra::copy3(lines[contact_surfs[m].index].norm, tmp);
            if (nsidek == OPPOSITE_SIDE)
              MathExtra::negate3(tmp);
          } else {
            // unconnected surf, use point
            MathExtra::copy3(contact_surfs[m].r, tmp);
          }
          MathExtra::norm3(tmp);
          for (a = 0; a < 3; a++) {
            xc[a] += tmp[a] * overlap;
            // correct omegac, vc
          }
        }
        MathExtra::norm3(xc);
        MathExtra::scale3(radi - max_overlap, xc);
        for (a = 0; a < 3; a++)
          xc[a] = x[i][a] - xc[a];
      } else {
        k = contact_surfs[n].index;
        if (hidden_contacts->find(k) != hidden_contacts->end())
          skip = 1;

        if (contact_surfs[n].ignore_pt) {
          for (a = 0; a < 3; a++)
            xc[a] = x[i][a] - lines[k].norm[a] * (radi - max_overlap);
        } else {
          for (a = 0; a < 3; a++)
            xc[a] = x[i][a] - contact_surfs[n].r[a];
        }
      }

      // Go to next contact if this one is obscured by a closer surf
      if (skip) continue;

      MathExtra::zero3(vc);
      MathExtra::zero3(omegac);
      // TODO correct velocity at contact point, from before:
      //j = contact_surfs[m].index
      //ds[0] = contact[0] - xsurf[j][0];
      //ds[1] = contact[1] - xsurf[j][1];
      //ds[2] = contact[2] - xsurf[j][2];
      //v_contact[0] = vsurf[j][0] + (omegasurf[j][1] * ds[2] - omegasurf[j][2] * ds[1]);
      //v_contact[1] = vsurf[j][1] + (omegasurf[j][2] * ds[0] - omegasurf[j][0] * ds[2]);
      //v_contact[2] = vsurf[j][2] + (omegasurf[j][0] * ds[1] - omegasurf[j][1] * ds[0]);

      model->xj = xc;
      model->vj = vc;
      model->omegaj = omegac;

      if (use_history) model->touch = touch[jj];

      // guaranteed in contact, but need to calculate intermediate variables
      touch_flag = model->check_contact();

      if (use_history) {
        jj = contact_surfs[n].neigh_index;

        // Check if another flat contact has a stored history
        if (touch[jj] != 1) {
          for (it = 0; it < flat_surfs->size(); it++) {
            m = (*flat_surfs)[it];
            jjtmp = contact_surfs[m].neigh_index;
            if (touch[jjtmp] == 1);
              jj = jjtmp;
          }
        }

        touch[jj] = 1;
        history = &allhistory[size_history * jj];
        model->history = history;
      }

      model->calculate_forces();

      // Sychronize history across flat contacts
      //   can be arbitrary if not all connected flat surfaces are mutually flat
      //   e.g. a hair pin turn where surfs on either end of the 'U' are not flat
      if (use_history) {
        for (it = 0; it < flat_surfs->size(); it++) {
          m = (*flat_surfs)[it];
          jjtmp = contact_surfs[m].neigh_index;
          if (jj != jjtmp) {
            touch[jjtmp] = 1;
            for (k = 0; k < size_history; k++)
              allhistory[size_history * jjtmp + k] = history[k];
          }
        }
      }

      forces = model->forces;
      torquesi = model->torquesi;

      add3(f[i], forces, f[i]);
      add3(torque[i], torquesi, torque[i]);
      if (heat_flag) heatflow[i] += model->dq;
    }
  }

  delete processed_contacts;
  delete contacts_map;
  delete flat_surfs;
}

/* ----------------------------------------------------------------------
   process fix_modify commands specific to fix surface/global
   move and type/region
------------------------------------------------------------------------- */

int FixSurfaceGlobal::modify_param(int narg, char **arg)
{
  // move keyword

  if (strcmp(arg[0],"move") == 0) {
    if (narg < 3) error->all(FLERR,"Illegal fix_modify move command");

    int lo,hi;
    int *stypes = new int[maxsurftype+1];
    for (int i = 1; i <= maxsurftype; i++) stypes[i] = 0;

    auto fields = Tokenizer(arg[1], ",").as_vector();
    for (int ifield = 0; ifield < fields.size(); ifield++) {
      utils::bounds(FLERR, fields[ifield], 1, maxsurftype, lo, hi, error);
      for (int i = lo; i <= hi; i++) stypes[i] = 1;
    }

    if (strcmp(arg[2],"none") == 0) {
      int count = 0;
      for (int itype = 1; itype <= maxsurftype; itype++) {
        if (type2motion[itype] < 0) continue;
        int imotion = type2motion[itype];
        motions[imotion].active = 0;
        type2motion[itype] = -1;

        // set vsurf and omegasurf to zero for itype surfs

        int stype;
        for (int i = 0; i < nsurf; i++) {
          if (dimension == 2) stype = lines[i].type;
          else stype = tris[i].type;
          if (stype == itype) {
            vsurf[i][0] = vsurf[i][1] = vsurf[i][2] = 0.0;
            omegasurf[i][0] = omegasurf[i][1] = omegasurf[i][2] = 0.0;
            count++;
          }
        }
      }

      // error check if any surf now assigned to inactive motion
      // b/c specification of types for "none" was incomplete

      int imotion;

      count = 0;
      if (dimension == 2) {
        for (int i = 0; i < nlines; i++) {
          imotion = type2motion[lines[i].type];
          if (imotion < 0) continue;
          if (!motions[imotion].active) count++;
        }
      } else {
        for (int i = 0; i < ntris; i++) {
          imotion = type2motion[tris[i].type];
          if (imotion < 0) continue;
          if (!motions[imotion].active) count++;
        }
      }

      if (count)
        error->all(FLERR,fmt::format("Fix_modify move none left {} surfs assigned inactive motion",count));

      // stats

      utils::logmesg(lmp,"Fix_modify move:\n");
      utils::logmesg(lmp,fmt::format("  turned off motion for {} surfs\n",count));

      // reset anymove and anymove_variable
      // if no anymove, deallocate memory and turn off INITIAL_INTEGRATE

      anymove = 0;
      for (int i = 1; i <= maxsurftype; i++)
        if (type2motion[i] >= 0) anymove = 1;

      anymove_variable = 0;
      for (int i = 0; i < nmotion; i++)
        if (motions[i].active && motions[i].mstyle == VARIABLE)
          anymove_variable = 1;

      if (!anymove) {
        memory->destroy(points_lastneigh);
        memory->destroy(points_original);
        memory->destroy(xsurf_original);
        memory->destroy(pointmove);
        points_lastneigh = nullptr;
        points_original = nullptr;
        xsurf_original = nullptr;
        pointmove = nullptr;

        int ifix = modify->find_fix(id);
        modify->fmask[ifix] &= ~INITIAL_INTEGRATE;
        force_reneighbor = 0;
        next_reneighbor = -1;
      }

      delete [] stypes;
      return 3;
    }

    // new motion operation
    // re-use an inactive motion or add a new motion to list

    int imotion;
    for (imotion = 0; imotion < nmotion; imotion++)
      if (!motions[imotion].active) break;

    if (imotion == nmotion) {
      if (nmotion == maxmotion) {
        maxmotion += DELTAMOTION;
        motions = (Motion *)
          memory->srealloc(motions,maxmotion*sizeof(Motion),"surface/global:motion");
      }
      nmotion++;
    }

    motions[imotion].active = 0;

    // use stypes to set type2motion

    for (int itype = 1; itype <= maxsurftype; itype++)
      if (stypes[itype]) type2motion[itype] = imotion;

    // if first motion, allocate points and surf memory

    if (!anymove) {
      memory->create(points_lastneigh,npoints,3,"surface/global:points_lastneigh");
      memory->create(points_original,npoints,3,"surface/global:points_original");
      memory->create(xsurf_original,nsurf,3,"surface/global:xsurf_original");
      memory->create(pointmove,npoints,"surface/global:pointmove");
    }

    anymove = 1;

    // parse additional move style arguments

    int styleargs = modify_param_move(&motions[imotion],narg-2,&arg[2]);
    int mstyle = motions[imotion].mstyle;
    if (mstyle == VARIABLE) anymove_variable = 1;
    motions[imotion].time_origin = update->ntimestep;

    int ifix = modify->find_fix(id);
    modify->fmask[ifix] |= INITIAL_INTEGRATE;

    force_reneighbor = 1;
    next_reneighbor = -1;

    // intialize points and surfs in stypes

    int itype,p1,p2,p3;
    double omega;
    double *runit;
    int count = 0;

    for (int i = 0; i < nsurf; i++) {
      if (dimension == 2) itype = lines[i].type;
      else itype = tris[i].type;
      if (!stypes[itype]) continue;
      count++;

      if (dimension == 2) {
        p1 = lines[i].p1;
        p2 = lines[i].p2;
        points_lastneigh[p1][0] = points_original[p1][0] = points[p1].x[0];
        points_lastneigh[p1][1] = points_original[p1][1] = points[p1].x[1];
        points_lastneigh[p1][2] = points_original[p1][2] = points[p1].x[2];
        points_lastneigh[p2][0] = points_original[p2][0] = points[p2].x[0];
        points_lastneigh[p2][1] = points_original[p2][1] = points[p2].x[1];
        points_lastneigh[p2][2] = points_original[p2][2] = points[p2].x[2];
      } else {
        p1 = tris[i].p1;
        p2 = tris[i].p2;
        p3 = tris[i].p3;
        points_lastneigh[p1][0] = points_original[p1][0] = points[p1].x[0];
        points_lastneigh[p1][1] = points_original[p1][1] = points[p1].x[1];
        points_lastneigh[p1][2] = points_original[p1][2] = points[p1].x[2];
        points_lastneigh[p2][0] = points_original[p2][0] = points[p2].x[0];
        points_lastneigh[p2][1] = points_original[p2][1] = points[p2].x[1];
        points_lastneigh[p2][2] = points_original[p2][2] = points[p2].x[2];
        points_lastneigh[p3][0] = points_original[p3][0] = points[p3].x[0];
        points_lastneigh[p3][1] = points_original[p3][1] = points[p3].x[1];
        points_lastneigh[p3][2] = points_original[p3][2] = points[p3].x[2];
      }

      xsurf_original[i][0] = xsurf[i][0];
      xsurf_original[i][1] = xsurf[i][1];
      xsurf_original[i][2] = xsurf[i][2];

      if (mstyle == ROTATE || mstyle == TRANSROT) {
        omega = motions[imotion].omega;
        runit = motions[imotion].unit;
        omegasurf[i][0] = omega*runit[0];
        omegasurf[i][1] = omega*runit[1];
        omegasurf[i][2] = omega*runit[2];
      }
    }

    utils::logmesg(lmp,"Fix_modify move:\n");
    utils::logmesg(lmp,fmt::format("  turned on motion for {} surfs\n",count));

    delete [] stypes;
    return 2 + styleargs;
  }

  // type/region keyword

  if (strcmp(arg[0],"type/region") == 0) {
    if (narg < 3) error->all(FLERR,"Illegal fix_modify command");

    int stype = utils::inumeric(FLERR,arg[1],false,lmp);
    if (stype <= 0 || stype > maxsurftype)
      error->all(FLERR,"Invalid fix_modify type/region surf type");

    auto region = domain->get_region_by_id(arg[2]);
    if (!region) error->all(FLERR,"Fix_modify type/region region {} does not exist", arg[2]);

    int count = 0;
    if (dimension == 2) {
      for (int i = 0; i < nlines; i++)
        if (region->match(xsurf[i][0],xsurf[i][1],xsurf[i][2])) {
          lines[i].type = stype;
          count++;
        }
    } else {
      for (int i = 0; i < ntris; i++)
        if (region->match(xsurf[i][0],xsurf[i][1],xsurf[i][2])) {
          tris[i].type = stype;
          count++;
        }
    }

    utils::logmesg(lmp,"Fix_modify type/region:\n");
    utils::logmesg(lmp,fmt::format("  {} surfs assigned to type {}\n",count,stype));

    return 3;
  }

  // keyword not recognized

  return 0;
}

/* ---------------------------------------------------------------------- */

int FixSurfaceGlobal::modify_param_move(Motion *motion, int narg, char **arg)
{
  if (strcmp(arg[0],"linear") == 0) {
    if (narg < 4) error->all(FLERR,"Illegal fix_modify move command");
    motion->mstyle = LINEAR;

    if (strcmp(arg[1], "NULL") == 0) motion->vxflag = 0;
    else {
      motion->vxflag = 1;
      motion->vx = utils::numeric(FLERR, arg[4], false, lmp);
    }
    if (strcmp(arg[2], "NULL") == 0) motion->vyflag = 0;
    else {
      motion->vyflag = 1;
      motion->vy = utils::numeric(FLERR, arg[2], false, lmp);
    }
    if (strcmp(arg[3], "NULL") == 0) motion->vzflag = 0;
    else {
      motion->vzflag = 1;
      motion->vz = utils::numeric(FLERR, arg[3], false, lmp);
    }

    if (dimension == 2)
      if (motion->vzflag && (motion->vz != 0.0))
        error->all(FLERR,"Fix_modify move cannot set linear z motion for 2d problem");

    return 4;
  }

  if (strcmp(arg[0],"wiggle") == 0) {
    if (narg < 5) error->all(FLERR,"Illegal fix_modify move command");
    motion->mstyle = WIGGLE;

    if (strcmp(arg[1], "NULL") == 0) motion->axflag = 0;
    else {
      motion->axflag = 1;
      motion->ax = utils::numeric(FLERR, arg[4], false, lmp);
    }
    if (strcmp(arg[2], "NULL") == 0) motion->ayflag = 0;
    else {
      motion->ayflag = 1;
      motion->ay = utils::numeric(FLERR, arg[2], false, lmp);
    }
    if (strcmp(arg[3], "NULL") == 0) motion->azflag = 0;
    else {
      motion->azflag = 1;
      motion->az = utils::numeric(FLERR, arg[3], false, lmp);
    }

    if (dimension == 2)
      if (motion->azflag && (motion->az != 0.0))
        error->all(FLERR,"Fix_modify move cannot set wiggle z motion for 2d problem");

    motion->period = utils::numeric(FLERR, arg[7], false, lmp);
    if (motion->period <= 0.0) error->all(FLERR, "Illegal fix_modify move command");
    motion->omega = MY_2PI / motion->period;

    return 5;
  }

  if (strcmp(arg[0],"rotate") == 0) {
    if (narg < 8) error->all(FLERR,"Illegal fix_modify move command");
    motion->mstyle = ROTATE;

    motion->point[0] = utils::numeric(FLERR,arg[1],false,lmp);
    motion->point[1] = utils::numeric(FLERR,arg[2],false,lmp);
    motion->point[2] = utils::numeric(FLERR,arg[3],false,lmp);

    motion->axis[0] = utils::numeric(FLERR,arg[4],false,lmp);
    motion->axis[1] = utils::numeric(FLERR,arg[5],false,lmp);
    motion->axis[2] = utils::numeric(FLERR,arg[6],false,lmp);

    if (dimension == 2)
      if (motion->axis[0] != 0.0 || motion->axis[1] != 0.0)
        error->all(FLERR,"Fix_modify move cannot rotate around "
                   "non z-axis for 2d problem");

    motion->period = utils::numeric(FLERR,arg[7],false,lmp);
    if (motion->period <= 0.0) error->all(FLERR,"Illegal fix_modify move command");

    motion->omega = MY_2PI / motion->period;

    // runit = unit vector along rotation axis

    double len = MathExtra::len3(motion->axis);
    if (len == 0.0)
      error->all(FLERR,"Fix_modify move zero length rotation vector");
    MathExtra::normalize3(motion->axis,motion->unit);

    return 8;
  }

  if (strcmp(arg[0],"transrot") == 0) {
    if (narg < 11) error->all(FLERR,"Illegal fix_modify move command");
    motion->mstyle = TRANSROT;

    error->all(FLERR,
               "Fix_modify move transrot not yet supported for fix surface/global");

    motion->vxflag = motion->vyflag = motion->vzflag = 1;
    motion->vx = utils::numeric(FLERR, arg[1], false, lmp);
    motion->vy = utils::numeric(FLERR, arg[2], false, lmp);
    motion->vz = utils::numeric(FLERR, arg[3], false, lmp);

    motion->point[0] = utils::numeric(FLERR,arg[4],false,lmp);
    motion->point[1] = utils::numeric(FLERR,arg[5],false,lmp);
    motion->point[2] = utils::numeric(FLERR,arg[6],false,lmp);

    motion->axis[0] = utils::numeric(FLERR,arg[7],false,lmp);
    motion->axis[1] = utils::numeric(FLERR,arg[8],false,lmp);
    motion->axis[2] = utils::numeric(FLERR,arg[9],false,lmp);

    if (dimension == 2) {
      if (motion->vzflag && (motion->vz != 0.0))
        error->all(FLERR,"Fix_modify move cannot set linear z motion for 2d problem");
      if (motion->axis[0] != 0.0 || motion->axis[1] != 0.0)
        error->all(FLERR,"Fix_modify move cannot rotate around "
                   "non z-axis for 2d problem");
    }

    motion->period = utils::numeric(FLERR,arg[10],false,lmp);
    if (motion->period <= 0.0) error->all(FLERR,"Illegal fix_modify move command");

    motion->omega = MY_2PI / motion->period;

    // runit = unit vector along rotation axis

    double len = MathExtra::len3(motion->axis);
    if (len == 0.0)
      error->all(FLERR,"Fix_modify move zero length rotation vector");
    MathExtra::normalize3(motion->axis,motion->unit);

    return 11;
  }

  if (strcmp(arg[0],"variable") == 0) {
    if (narg < 7) error->all(FLERR,"Illegal fix_modify move command");
    motion->mstyle = VARIABLE;

    if (strcmp(arg[1], "NULL") == 0)
      motion->xvarstr = nullptr;
    else if (utils::strmatch(arg[1], "^v_")) {
      motion->xvarstr = utils::strdup(arg[1] + 2);
    } else
      error->all(FLERR, "Illegal fix_modify move command");
    if (strcmp(arg[2], "NULL") == 0)
      motion->yvarstr = nullptr;
    else if (utils::strmatch(arg[2], "^v_")) {
      motion->yvarstr = utils::strdup(arg[2] + 2);
    } else
      error->all(FLERR, "Illegal fix_modify move command");
    if (strcmp(arg[3], "NULL") == 0)
      motion->zvarstr = nullptr;
    else if (utils::strmatch(arg[3], "^v_")) {
      motion->zvarstr = utils::strdup(arg[3] + 2);
    } else
      error->all(FLERR, "Illegal fix_modify move command");


    if (strcmp(arg[4], "NULL") == 0)
      motion->vxvarstr = nullptr;
    else if (utils::strmatch(arg[4], "^v_")) {
      motion->vxvarstr = utils::strdup(arg[4] + 2);
    } else
      error->all(FLERR, "Illegal fix_modify move command");
    if (strcmp(arg[5], "NULL") == 0)
      motion->vyvarstr = nullptr;
    else if (utils::strmatch(arg[5], "^v_")) {
      motion->vyvarstr = utils::strdup(arg[5] + 2);
    } else
      error->all(FLERR, "Illegal fix_modify move command");
    if (strcmp(arg[6], "NULL") == 0)
      motion->vzvarstr = nullptr;
    else if (utils::strmatch(arg[6], "^v_")) {
      motion->vzvarstr = utils::strdup(arg[6] + 2);
    } else
      error->all(FLERR, "Illegal fix_modify move command");

    if (dimension == 2 && (motion->zvarstr || motion->vzvarstr))
      error->all(FLERR, "Fix_modify move cannot define z or vz variable for 2d problem");

    return 7;
  }

  error->all(FLERR,"Fix_modify move style not recognized");

  return 0;
}

/* ---------------------------------------------------------------------- */

void FixSurfaceGlobal::reset_dt()
{
  /*
  if (mstyle != NONE)
    error->all(FLERR,"Resetting timestep size is not allowed with "
               "fix surface/global motion");
  */
}

/* ----------------------------------------------------------------------
   memory usage per-surf data and neighbor list
------------------------------------------------------------------------- */

double FixSurfaceGlobal::memory_usage()
{
  double bytes = 0.0;

  // points, lines, tris and connect2d/3d

  bytes += npoints*sizeof(Point);
  if (dimension == 2) {
    bytes += nlines*sizeof(Line);
    bytes += nlines*sizeof(Connect2d);
  } else if (dimension == 3) {
    bytes += ntris*sizeof(Tri);
    bytes += ntris*sizeof(Connect3d);
  }

  bytes += memory->usage(xsurf,nsurf,3);
  bytes += memory->usage(vsurf,nsurf,3);
  bytes += memory->usage(omegasurf,nsurf,3);
  bytes += memory->usage(radsurf,nsurf);

  if (anymove) {
    bytes += memory->usage(points_lastneigh,npoints,3);
    bytes += memory->usage(points_original,npoints,3);
    bytes += memory->usage(xsurf_original,nsurf,3);
    bytes += memory->usage(pointmove,nsurf);
  }

  if (mass_rigid) bytes += memory->usage(mass_rigid,atom->nmax);

  if (imax) {
    bytes += memory->usage(imflag,nsurf);
    if (dimension == 2) bytes += memory->usage(imdata,nsurf,7);
    else if (dimension == 3) bytes += memory->usage(imdata,nsurf,10);
  }

  // ragged connectivity arrays

  if (dimension == 2) {
    int np = 0;
    for (int i = 0; i < nlines; i++) np += connect2d[i].np1 + connect2d[i].np2;
    bytes += 4*np * sizeof(int);
  } else if (dimension == 3) {
    int ne = 0;
    for (int i = 0; i < ntris; i++) ne += connect3d[i].ne1 + connect3d[i].ne2 + connect3d[i].ne3;
    bytes += 4*ne * sizeof(int);
    int nc = 0;
    for (int i = 0; i < ntris; i++) nc += connect3d[i].nc1 + connect3d[i].nc2 + connect3d[i].nc3;
    bytes += 2*nc * sizeof(int);
  }

  // neighbor list

  bytes += list->memory_usage();

  return bytes;
}

/* ----------------------------------------------------------------------
   extract neighbor lists
------------------------------------------------------------------------- */

void *FixSurfaceGlobal::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str,"list") == 0) return list;
  else if (strcmp(str,"listhistory") == 0) return listhistory;
  return nullptr;
}

/* ---------------------------------------------------------------------- */

int FixSurfaceGlobal::image(int *&ivec, double **&darray)
{
  int n;
  double *p1,*p2,*p3;

  if (dimension == 2) {
    n = nlines;

    if (imax == 0) {
      imax = n;
      memory->create(imflag,imax,"surface/global:imflag");
      memory->create(imdata,imax,7,"surface/global:imflag");
    }

    for (int i = 0; i < n; i++) {
      p1 = points[lines[i].p1].x;
      p2 = points[lines[i].p2].x;

      imflag[i] = LINE;
      imdata[i][0] = lines[i].type;
      imdata[i][1] = p1[0];
      imdata[i][2] = p1[1];
      imdata[i][3] = p1[2];
      imdata[i][4] = p2[0];
      imdata[i][5] = p2[1];
      imdata[i][6] = p2[2];
    }

  } else {
    n = ntris;

    if (imax == 0) {
      imax = n;
      memory->create(imflag,imax,"surface/global:imflag");
      memory->create(imdata,imax,10,"surface/global:imflag");
    }

    for (int i = 0; i < n; i++) {
      p1 = points[tris[i].p1].x;
      p2 = points[tris[i].p2].x;
      p3 = points[tris[i].p3].x;

      imflag[i] = TRI;
      imdata[i][0] = tris[i].type;
      imdata[i][1] = p1[0];
      imdata[i][2] = p1[1];
      imdata[i][3] = p1[2];
      imdata[i][4] = p2[0];
      imdata[i][5] = p2[1];
      imdata[i][6] = p2[2];
      imdata[i][7] = p3[0];
      imdata[i][8] = p3[1];
      imdata[i][9] = p3[2];
    }
  }

  ivec = imflag;
  darray = imdata;
  return n;
}

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// initializiation of surfs
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------

/* ----------------------------------------------------------------------
   extract lines or tris from a molecule template ID for one or more molecules
   concatenate into single list of points and lines or tris
   identify unique points using hash
------------------------------------------------------------------------- */

void FixSurfaceGlobal::extract_from_molecule(char *molID,
                                             std::map<std::tuple<double,double,double>,int> *hash)
{
  int imol = atom->find_molecule(molID);
  if (imol == -1)
    error->all(FLERR,"Molecule template ID for fix surface/global does not exist");

  // loop over one or more molecules in molID

  Molecule **onemols = &atom->molecules[imol];
  int nmol = onemols[0]->nset;

  for (int m = 0; m < nmol; m++) {
    if (dimension == 2)
      if (onemols[m]->lineflag == 0)
        error->all(FLERR,"Fix surface/global molecule must have lines");
    if (dimension == 3)
      if (onemols[m]->triflag == 0)
        error->all(FLERR,"Fix surface/global molecule must have triangles");

    int nl = onemols[m]->nlines;
    int nt = onemols[m]->ntris;

    nlines += nl;
    ntris += nt;
    lines = (Line *) memory->srealloc(lines,nlines*sizeof(Line),
                                      "surface/global:lines");
    tris = (Tri *) memory->srealloc(tris,ntris*sizeof(Tri),
                                    "surface/global:tris");

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
        if (hash->find(key) == hash->end()) {
          if (npoints == maxpoints) {
            maxpoints += DELTA;
            points = (Point *) memory->srealloc(points,maxpoints*sizeof(Point),
                                                "surface/global:points");
          }
          (*hash)[key] = npoints;
          points[npoints].x[0] = epts[i][0];
          points[npoints].x[1] = epts[i][1];
          points[npoints].x[2] = 0.0;
          lines[iline].p1 = npoints;
          npoints++;
        } else lines[iline].p1 = (*hash)[key];

        key = std::make_tuple(epts[i][2],epts[i][3],0.0);
        if (hash->find(key) == hash->end()) {
          if (npoints == maxpoints) {
            maxpoints += DELTA;
            points = (Point *) memory->srealloc(points,maxpoints*sizeof(Point),
                                                "surface/global:points");
          }
          (*hash)[key] = npoints;
          points[npoints].x[0] = epts[i][2];
          points[npoints].x[1] = epts[i][3];
          points[npoints].x[2] = 0.0;
          lines[iline].p2 = npoints;
          npoints++;
        } else lines[iline].p2 = (*hash)[key];

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
        if (hash->find(key) == hash->end()) {
          if (npoints == maxpoints) {
            maxpoints += DELTA;
            points = (Point *) memory->srealloc(points,maxpoints*sizeof(Point),
                                                "surface/global:points");
          }
          (*hash)[key] = npoints;
          points[npoints].x[0] = cpts[i][0];
          points[npoints].x[1] = cpts[i][1];
          points[npoints].x[2] = cpts[i][2];
          tris[itri].p1 = npoints;
          npoints++;
        } else tris[itri].p1 = (*hash)[key];

        key = std::make_tuple(cpts[i][3],cpts[i][4],cpts[i][5]);
        if (hash->find(key) == hash->end()) {
          if (npoints == maxpoints) {
            maxpoints += DELTA;
            points = (Point *) memory->srealloc(points,maxpoints*sizeof(Point),
                                                "surface/global:points");
          }
          (*hash)[key] = npoints;
          points[npoints].x[0] = cpts[i][3];
          points[npoints].x[1] = cpts[i][4];
          points[npoints].x[2] = cpts[i][5];
          tris[itri].p2 = npoints;
          npoints++;
        } else tris[itri].p2 = (*hash)[key];

        key = std::make_tuple(cpts[i][6],cpts[i][7],cpts[i][8]);
        if (hash->find(key) == hash->end()) {
          if (npoints == maxpoints) {
            maxpoints += DELTA;
            points = (Point *) memory->srealloc(points,maxpoints*sizeof(Point),
                                                "surface/global:points");
          }
          (*hash)[key] = npoints;
          points[npoints].x[0] = cpts[i][6];
          points[npoints].x[1] = cpts[i][7];
          points[npoints].x[2] = cpts[i][8];
          tris[itri].p3 = npoints;
          npoints++;
        } else tris[itri].p3 = (*hash)[key];

        itri++;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   extract triangles from an STL file, can be text or binary
   concatenate into single list of points and tris
   identify unique points using hash
------------------------------------------------------------------------- */

void FixSurfaceGlobal::extract_from_stlfile(char *filename, int stype,
                                            std::map<std::tuple<double,double,double>,int> *hash)
{
  if (dimension == 2)
    error->all(FLERR,"Fix surface/global cannot use an STL file for 2d simulations");

  // read tris from STL file
  // stltris = tri coords internal to STL reader

  STLReader *stl = new STLReader(lmp);
  double **stltris;
  ntris = stl->read_file(filename,stltris);

  tris = (Tri *) memory->srealloc(tris,ntris*sizeof(Tri),
                                  "surface/global:tris");

  // loop over STL tris
  // populate points and tris data structs
  // for each tri: set molID = 1 and type = stype

  for (int itri = 0; itri < ntris; itri++) {
    tris[itri].mol = 1;
    tris[itri].type = stype;

    auto key = std::make_tuple(stltris[itri][0],stltris[itri][1],stltris[itri][2]);
    if (hash->find(key) == hash->end()) {
      if (npoints == maxpoints) {
        maxpoints += DELTA;
        points = (Point *) memory->srealloc(points,maxpoints*sizeof(Point),
                                            "surface/global:points");
      }
      (*hash)[key] = npoints;
      points[npoints].x[0] = stltris[itri][0];
      points[npoints].x[1] = stltris[itri][1];
      points[npoints].x[2] = stltris[itri][2];
      tris[itri].p1 = npoints;
      npoints++;
    } else tris[itri].p1 = (*hash)[key];

    key = std::make_tuple(stltris[itri][3],stltris[itri][4],stltris[itri][5]);
    if (hash->find(key) == hash->end()) {
      if (npoints == maxpoints) {
        maxpoints += DELTA;
        points = (Point *) memory->srealloc(points,maxpoints*sizeof(Point),
                                            "surface/global:points");
      }
      (*hash)[key] = npoints;
      points[npoints].x[0] = stltris[itri][3];
      points[npoints].x[1] = stltris[itri][4];
      points[npoints].x[2] = stltris[itri][5];
      tris[itri].p2 = npoints;
      npoints++;
    } else tris[itri].p2 = (*hash)[key];

    key = std::make_tuple(stltris[itri][6],stltris[itri][7],stltris[itri][8]);
    if (hash->find(key) == hash->end()) {
      if (npoints == maxpoints) {
        maxpoints += DELTA;
        points = (Point *) memory->srealloc(points,maxpoints*sizeof(Point),
                                            "surface/global:points");
      }
      (*hash)[key] = npoints;
      points[npoints].x[0] = stltris[itri][6];
      points[npoints].x[1] = stltris[itri][7];
      points[npoints].x[2] = stltris[itri][8];
      tris[itri].p3 = npoints;
      npoints++;
    } else tris[itri].p3 = (*hash)[key];
  }

  // delete STL reader

  delete stl;
}

/* ----------------------------------------------------------------------
   error checks on lines
   no zero-length lines and no duplicate lines
------------------------------------------------------------------------- */

void FixSurfaceGlobal::check2d()
{
  // check for zero length lines
  // use coords of points, not indices
  // since (p1,p2) can be zero-length even if p1 != p2

  double *pt1,*pt2;

  int flag = 0;
  for (int i = 0; i < nlines; i++) {
    pt1 = points[lines[i].p1].x;
    pt2 = points[lines[i].p2].x;
    if (pt1[0] == pt2[0] && pt1[1] == pt2[1]) flag++;
  }

  if (flag)
    error->all(FLERR,fmt::format("Fix surface/global defines {} zero-length lines",flag));

  // check for duplicate lines
  // (p1,p2) is duplicate of any line with same 2 endpoints
  // hash = map of lines
  //   key = <p1,p2> indices of 2 points
  //   value = 1 if already appeared

  int p1,p2;
  std::map<std::tuple<int,int>,int> hash;

  flag = 0;
  for (int i = 0; i < nlines; i++) {
    p1 = lines[i].p1;
    p2 = lines[i].p2;

    auto key1 = std::make_tuple(p1,p2);
    auto key2 = std::make_tuple(p2,p1);

    if (hash.find(key1) == hash.end() && hash.find(key2) == hash.end()) hash[key1] = 1;
    else flag++;
  }

  if (flag)
    error->all(FLERR,fmt::format("Fix surface/global defines {} duplicate lines",flag));
}


/* ----------------------------------------------------------------------
   error checks on tris
   no tris with a zero-length edge
   no duplicate tris
------------------------------------------------------------------------- */

void FixSurfaceGlobal::check3d()
{
  // check for zero length tri edges
  // use coords of points, not indices
  // since (p1,p2) can be zero-length even if p1 != p2

  double *pt1,*pt2,*pt3;

  int flag = 0;
  for (int i = 0; i < ntris; i++) {
    pt1 = points[tris[i].p1].x;
    pt2 = points[tris[i].p2].x;
    pt3 = points[tris[i].p3].x;

    if (pt1[0] == pt2[0] && pt1[1] == pt2[1]) flag++;
    if (pt2[0] == pt3[0] && pt2[1] == pt3[1]) flag++;
    if (pt3[0] == pt1[0] && pt3[1] == pt1[1]) flag++;
  }

  if (flag)
    error->all(FLERR,fmt::format("Fix surface/global defines {} zero-length triangle edges",flag));

  // check for duplicate tris
  // (p1,p2,p3) of any tri with same 3 corner points
  // hash = map of tris
  //   key = <p1,p2,p3> indices of 3 points
  //   value = 1 if already appeared

  int p1,p2,p3;
  std::map<std::tuple<int,int,int>,int> hash;

  flag = 0;
  for (int i = 0; i < ntris; i++) {
    p1 = tris[i].p1;
    p2 = tris[i].p2;
    p3 = tris[i].p2;

    auto key1 = std::make_tuple(p1,p2,p3);
    auto key2 = std::make_tuple(p1,p3,p2);
    auto key3 = std::make_tuple(p2,p1,p3);
    auto key4 = std::make_tuple(p2,p3,p1);
    auto key5 = std::make_tuple(p3,p1,p2);
    auto key6 = std::make_tuple(p3,p2,p1);

    if (hash.find(key1) == hash.end() && hash.find(key2) == hash.end() &&
        hash.find(key3) == hash.end() && hash.find(key4) == hash.end() &&
        hash.find(key5) == hash.end() && hash.find(key6) == hash.end()) hash[key1] = 1;
    else flag++;
  }

  if (flag)
    error->all(FLERR,fmt::format("Fix surface/global defines {} duplicate triangles",flag));
}

/* ----------------------------------------------------------------------
   issue warning if any connected lines/tris have different molIDs
   molIDs are not used now, but may be in future
   an inter-connected set of surfs should have a single molID
------------------------------------------------------------------------- */

void FixSurfaceGlobal::check_molecules()
{
  int i,j,m,imol,flag;

  if (dimension == 2) {
    int *neigh_p1,*neigh_p2;
    flag = 0;
    for (i = 0; i < nlines; i++) {
      imol = lines[i].mol;
      neigh_p1 = connect2d[i].neigh_p1;
      neigh_p2 = connect2d[i].neigh_p2;
      for (m = 0; m < connect2d[i].np1; m++)
        if (imol != lines[neigh_p1[m]].mol) flag++;
      for (m = 0; m < connect2d[i].np2; m++)
        if (imol != lines[neigh_p2[m]].mol) flag++;
    }
    flag /= 2;
    if (flag && comm->me == 0)
      error->warning(FLERR,
                     "Fix surface/global endpoint connections between "
                     "different molecule IDs =",flag);

  } else {
    int *neigh_e1,*neigh_e2,*neigh_e3;
    int *neigh_c1,*neigh_c2,*neigh_c3;
    flag = 0;
    for (i = 0; i < ntris; i++) {
      imol = tris[i].mol;
      neigh_e1 = connect3d[i].neigh_e1;
      neigh_e2 = connect3d[i].neigh_e2;
      neigh_e3 = connect3d[i].neigh_e3;
      for (m = 0; m < connect3d[i].ne1; m++)
        if (imol != tris[neigh_e1[m]].mol) flag++;
      for (m = 0; m < connect3d[i].ne2; m++)
        if (imol != tris[neigh_e2[m]].mol) flag++;
      for (m = 0; m < connect3d[i].ne3; m++)
        if (imol != tris[neigh_e3[m]].mol) flag++;
      neigh_c1 = connect3d[i].neigh_c1;
      neigh_c2 = connect3d[i].neigh_c2;
      neigh_c3 = connect3d[i].neigh_c3;
      for (m = 0; m < connect3d[i].nc1; m++)
        if (imol != tris[neigh_c1[m]].mol) flag++;
      for (m = 0; m < connect3d[i].nc2; m++)
        if (imol != tris[neigh_c2[m]].mol) flag++;
      for (m = 0; m < connect3d[i].nc3; m++)
        if (imol != tris[neigh_c3[m]].mol) flag++;
    }
    flag /= 2;
    if (flag && comm->me == 0)
      error->warning(FLERR,
                     "Fix surface/global edge/corner connections between "
                     "different molecule IDs =",flag);
  }
}

/* ----------------------------------------------------------------------
   create and initialize Connect2d info for all lines
   this creates plines and connect2d data structs
------------------------------------------------------------------------- */

void FixSurfaceGlobal::connectivity2d()
{
  connect2d = (Connect2d *)
    memory->smalloc(nlines*sizeof(Connect2d),
                    "surface/global:connect2d");

  // setup line end point connectivity lists
  // counts = # of lines containing each end point (including self)
  // plines = ragged 2d array with indices of lines which contain each point

  int *counts;
  memory->create(counts,npoints,"surface/global:counts");

  for (int i = 0; i < npoints; i++) counts[i] = 0;

  for (int i = 0; i < nlines; i++) {
    counts[lines[i].p1]++;
    counts[lines[i].p2]++;
  }

  int **plines;
  memory->create_ragged(plines,npoints,counts,"surface/global:plines");

  for (int i = 0; i < npoints; i++) counts[i] = 0;

  for (int i = 0; i < nlines; i++) {
    plines[lines[i].p1][counts[lines[i].p1]++] = i;
    plines[lines[i].p2][counts[lines[i].p2]++] = i;
  }

  // p12_counts = # of lines connecting to endpoints p12 of each line
  // do NOT include self

  int *p1_counts,*p2_counts;
  memory->create(p1_counts,nlines,"surface/global:p1_counts");
  memory->create(p2_counts,nlines,"surface/global:p2_counts");

  for (int i = 0; i < nlines; i++) {
    p1_counts[i] = counts[lines[i].p1] - 1;
    p2_counts[i] = counts[lines[i].p2] - 1;
  }

  // allocate all ragged arrays which Connect2d will point to

  memory->create_ragged(neigh_p1,nlines,p1_counts,"surface/global:neigh_p1");
  memory->create_ragged(neigh_p2,nlines,p2_counts,"surface/global:neigh_p2");
  memory->create_ragged(pwhich_p1,nlines,p1_counts,"surface/global:pwhich_p1");
  memory->create_ragged(pwhich_p2,nlines,p2_counts,"surface/global:pwhich_p2");
  memory->create_ragged(nside_p1,nlines,p1_counts,"surface/global:nside_p1");
  memory->create_ragged(nside_p2,nlines,p2_counts,"surface/global:nside_p2");
  memory->create_ragged(aflag_p1,nlines,p1_counts,"surface/global:aflag_p1");
  memory->create_ragged(aflag_p2,nlines,p2_counts,"surface/global:aflag_p2");

  // set connect2d vector ptrs to rows of corresponding ragged arrays

  for (int i = 0; i < nlines; i++) {
    connect2d[i].np1 = p1_counts[i];
    if (connect2d[i].np1 == 0) {
      connect2d[i].neigh_p1 = nullptr;
      connect2d[i].pwhich_p1 = nullptr;
      connect2d[i].nside_p1 = nullptr;
      connect2d[i].aflag_p1 = nullptr;
    } else {
      connect2d[i].neigh_p1 = neigh_p1[i];
      connect2d[i].pwhich_p1 = pwhich_p1[i];
      connect2d[i].nside_p1 = nside_p1[i];
      connect2d[i].aflag_p1 = aflag_p1[i];
    }

    connect2d[i].np2 = p2_counts[i];
    if (connect2d[i].np2 == 0) {
      connect2d[i].neigh_p2 = nullptr;
      connect2d[i].pwhich_p2 = nullptr;
      connect2d[i].nside_p2 = nullptr;
      connect2d[i].aflag_p2 = nullptr;
    } else {
      connect2d[i].neigh_p2 = neigh_p2[i];
      connect2d[i].pwhich_p2 = pwhich_p2[i];
      connect2d[i].nside_p2 = nside_p2[i];
      connect2d[i].aflag_p2 = aflag_p2[i];
    }
  }

  // initialize connect2d neigh vectors for each end point of each line
  // do NOT include self

  int j,m;

  for (int i = 0; i < nlines; i++) {
    if (connect2d[i].np1) {
      j = 0;
      for (m = 0; m < counts[lines[i].p1]; m++) {
        if (plines[lines[i].p1][m] == i) continue;
        connect2d[i].neigh_p1[j] = plines[lines[i].p1][m];
        j++;
      }
    }
    if (connect2d[i].np2) {
      j = 0;
      for (m = 0; m < counts[lines[i].p2]; m++) {
        if (plines[lines[i].p2][m] == i) continue;
        connect2d[i].neigh_p2[j] = plines[lines[i].p2][m];
        j++;
      }
    }
  }

  // deallocate counts, plines, p12_counts

  memory->destroy(counts);
  memory->destroy(plines);
  memory->destroy(p1_counts);
  memory->destroy(p2_counts);

  // set connect2d pwhich/nside/aflag for each end point of each line
  // see fsg.h file for an explanation of each vector in Connect2d
  // aflag is based on dot and cross product of 2 connected line normals
  //   cross product is either along +z or -z direction

  double dotline,dotnorm;
  double *inorm,*jnorm;
  double icrossj[3];

  for (int i = 0; i < nlines; i++) {
    for (m = 0; m < connect2d[i].np1; m++) {
      j = connect2d[i].neigh_p1[m];

      inorm = lines[i].norm;
      jnorm = lines[j].norm;
      dotnorm = MathExtra::dot3(inorm,jnorm);

      if (lines[i].p1 == lines[j].p1) {
        connect2d[i].pwhich_p1[m] = 0;
        connect2d[i].nside_p1[m] = OPPOSITE_SIDE;
        if (dotnorm < -1.0+flatthresh) {
          connect2d[i].aflag_p1[m] = FLAT;
        } else {
          MathExtra::cross3(inorm,jnorm,icrossj);
          if (icrossj[2] > 0.0) connect2d[i].aflag_p1[m] = CONCAVE;
          else connect2d[i].aflag_p1[m] = CONVEX;
        }
      } else if (lines[i].p1 == lines[j].p2) {
        connect2d[i].pwhich_p1[m] = 1;
        connect2d[i].nside_p1[m] = SAME_SIDE;
        if (dotnorm > 1.0-flatthresh) {
          connect2d[i].aflag_p1[m] = FLAT;
        } else {
          MathExtra::cross3(inorm,jnorm,icrossj);
          if (icrossj[2] < 0.0) connect2d[i].aflag_p1[m] = CONCAVE;
          else connect2d[i].aflag_p1[m] = CONVEX;
        }
      }
    }

    for (m = 0; m < connect2d[i].np2; m++) {
      j = connect2d[i].neigh_p2[m];

      inorm = lines[i].norm;
      jnorm = lines[j].norm;
      dotnorm = MathExtra::dot3(inorm,jnorm);

      if (lines[i].p2 == lines[j].p1) {
        connect2d[i].pwhich_p2[m] = 0;
        connect2d[i].nside_p2[m] = SAME_SIDE;
        if (dotnorm > 1.0-flatthresh) {
          connect2d[i].aflag_p2[m] = FLAT;
        } else {
          MathExtra::cross3(inorm,jnorm,icrossj);
          if (icrossj[2] > 0.0) connect2d[i].aflag_p2[m] = CONCAVE;
          else connect2d[i].aflag_p2[m] = CONVEX;
        }
      } else if (lines[i].p2 == lines[j].p2) {
        connect2d[i].pwhich_p2[m] = 1;
        connect2d[i].nside_p2[m] = OPPOSITE_SIDE;
        if (dotnorm < -1.0+flatthresh) {
          connect2d[i].aflag_p2[m] = FLAT;
        } else {
          MathExtra::cross3(inorm,jnorm,icrossj);
          if (icrossj[2] < 0.0) connect2d[i].aflag_p2[m] = CONCAVE;
          else connect2d[i].aflag_p2[m] = CONVEX;
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   create and initialize Connect3d info for all triangles
   this creates etris/ctris and connect3d data structs
------------------------------------------------------------------------- */

void FixSurfaceGlobal::connectivity3d()
{
  int p1,p2,p3;

  connect3d = (Connect3d *)
    memory->smalloc(ntris*sizeof(Connect3d),
                    "surface/global:connect3d");

  // create hash = map of unique edges
  //   key = <p1,p2> indices of 2 points, in either order
  //   value = index of the unique edge (0 to Nedge-1)
  // tri2edges[i][j] = index of unique edge for tri I and edge J
  // nedges = total count of unique edges, stored for use elsewhere

  int **tri2edge;
  memory->create(tri2edge,ntris,3,"surfface/global::tri2edge");

  std::map<std::tuple<int,int>,int> hash;
  nedges = 0;

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
  memory->create(counts,nedges,"surface/global:count");

  for (int i = 0; i < nedges; i++) counts[i] = 0;

  for (int i = 0; i < ntris; i++) {
    counts[tri2edge[i][0]]++;
    counts[tri2edge[i][1]]++;
    counts[tri2edge[i][2]]++;
  }

  int **etris;
  memory->create_ragged(etris,nedges,counts,"surface/global:etris");

  for (int i = 0; i < nedges; i++) counts[i] = 0;

  for (int i = 0; i < ntris; i++) {
    etris[tri2edge[i][0]][counts[tri2edge[i][0]]++] = i;
    etris[tri2edge[i][1]][counts[tri2edge[i][1]]++] = i;
    etris[tri2edge[i][2]][counts[tri2edge[i][2]]++] = i;
  }

  // e123_counts = # of edges connecting to edges e123 of each tri
  // do NOT include self

  int *e1_counts,*e2_counts,*e3_counts;
  memory->create(e1_counts,ntris,"surface/global:e1_counts");
  memory->create(e2_counts,ntris,"surface/global:e2_counts");
  memory->create(e3_counts,ntris,"surface/global:e3_counts");

  for (int i = 0; i < ntris; i++) {
    e1_counts[i] = counts[tri2edge[i][0]] - 1;
    e2_counts[i] = counts[tri2edge[i][1]] - 1;
    e3_counts[i] = counts[tri2edge[i][2]] - 1;
  }

  // allocate all edge ragged arrays which Connect3d will point to

  memory->create_ragged(neigh_e1,ntris,e1_counts,"surface/global:neigh_e1");
  memory->create_ragged(neigh_e2,ntris,e2_counts,"surface/global:neigh_e2");
  memory->create_ragged(neigh_e3,ntris,e3_counts,"surface/global:neigh_e3");
  memory->create_ragged(ewhich_e1,ntris,e1_counts,"surface/global:ewhich_e1");
  memory->create_ragged(ewhich_e2,ntris,e2_counts,"surface/global:ewhich_e2");
  memory->create_ragged(ewhich_e3,ntris,e3_counts,"surface/global:ewhich_e3");
  memory->create_ragged(nside_e1,ntris,e1_counts,"surface/global:nside_e1");
  memory->create_ragged(nside_e2,ntris,e2_counts,"surface/global:nside_e2");
  memory->create_ragged(nside_e3,ntris,e3_counts,"surface/global:nside_e3");
  memory->create_ragged(aflag_e1,ntris,e1_counts,"surface/global:aflag_e1");
  memory->create_ragged(aflag_e2,ntris,e2_counts,"surface/global:aflag_e2");
  memory->create_ragged(aflag_e3,ntris,e3_counts,"surface/global:aflag_e3");

  // set connect3d edge vector ptrs to rows of corresponding edge ragged arrays

  for (int i = 0; i < ntris; i++) {
    connect3d[i].ne1 = e1_counts[i];
    if (connect3d[i].ne1 == 0) {
      connect3d[i].neigh_e1 = nullptr;
      connect3d[i].ewhich_e1 = nullptr;
      connect3d[i].nside_e1 = nullptr;
      connect3d[i].aflag_e1 = nullptr;
    } else {
      connect3d[i].neigh_e1 = neigh_e1[i];
      connect3d[i].ewhich_e1 = ewhich_e1[i];
      connect3d[i].nside_e1 = nside_e1[i];
      connect3d[i].aflag_e1 = aflag_e1[i];
    }

    connect3d[i].ne2 = e2_counts[i];
    if (connect3d[i].ne2 == 0) {
      connect3d[i].neigh_e2 = nullptr;
      connect3d[i].ewhich_e2 = nullptr;
      connect3d[i].nside_e2 = nullptr;
      connect3d[i].aflag_e2 = nullptr;
    } else {
      connect3d[i].neigh_e2 = neigh_e2[i];
      connect3d[i].ewhich_e2 = ewhich_e2[i];
      connect3d[i].nside_e2 = nside_e2[i];
      connect3d[i].aflag_e2 = aflag_e2[i];
    }

    connect3d[i].ne3 = e3_counts[i];
    if (connect3d[i].ne3 == 0) {
      connect3d[i].neigh_e3 = nullptr;
      connect3d[i].ewhich_e3 = nullptr;
      connect3d[i].nside_e3 = nullptr;
      connect3d[i].aflag_e3 = nullptr;
    } else {
      connect3d[i].neigh_e3 = neigh_e3[i];
      connect3d[i].ewhich_e3 = ewhich_e3[i];
      connect3d[i].nside_e3 = nside_e3[i];
      connect3d[i].aflag_e3 = aflag_e3[i];
    }
  }

  // initialize connect3d edge neigh vectors for each edge of each tri
  // do NOT include self

  int j,m;

  for (int i = 0; i < ntris; i++) {
    if (connect3d[i].ne1) {
      j = 0;
      for (m = 0; m < counts[tri2edge[i][0]]; m++) {
        if (etris[tri2edge[i][0]][m] == i) continue;
        connect3d[i].neigh_e1[j] = etris[tri2edge[i][0]][m];
        j++;
      }
    }
    if (connect3d[i].ne2) {
      j = 0;
      for (m = 0; m < counts[tri2edge[i][1]]; m++) {
        if (etris[tri2edge[i][1]][m] == i) continue;
        connect3d[i].neigh_e2[j] = etris[tri2edge[i][1]][m];
        j++;
      }
    }
    if (connect3d[i].ne3) {
      j = 0;
      for (m = 0; m < counts[tri2edge[i][2]]; m++) {
        if (etris[tri2edge[i][2]][m] == i) continue;
        connect3d[i].neigh_e3[j] = etris[tri2edge[i][2]][m];
        j++;
      }
    }
  }

  // deallocate counts, tri2edge, etris, e123_counts

  memory->destroy(counts);
  memory->destroy(tri2edge);
  memory->destroy(etris);
  memory->destroy(e1_counts);
  memory->destroy(e2_counts);
  memory->destroy(e3_counts);

  // set connect3d edge ewhich/nside/aflag for each edge of each tri
  // see fsg.h file for an explanation of each edge vector in Connect3d
  // aflag is based on dot and cross product of 2 connected tri normals
  //   cross product is either along itri edge or in opposite dir

  int jpfirst,jpsecond;
  double dotline,dotnorm;
  double *inorm,*jnorm;
  double icrossj[3],iedge[3];

  for (int i = 0; i < ntris; i++) {
    for (m = 0; m < connect3d[i].ne1; m++) {
      j = connect3d[i].neigh_e1[m];

      if (tris[i].p1 == tris[j].p1) jpfirst = 1;
      else if (tris[i].p1 == tris[j].p2) jpfirst = 2;
      else if (tris[i].p1 == tris[j].p3) jpfirst = 3;

      if (tris[i].p2 == tris[j].p1) jpsecond = 1;
      else if (tris[i].p2 == tris[j].p2) jpsecond = 2;
      else if (tris[i].p2 == tris[j].p3) jpsecond = 3;

      inorm = tris[i].norm;
      jnorm = tris[j].norm;
      dotnorm = MathExtra::dot3(inorm,jnorm);
      MathExtra::sub3(points[tris[i].p2].x,points[tris[i].p1].x,iedge);

      if ((jpfirst == 1 && jpsecond == 2) ||
          (jpfirst == 2 && jpsecond == 3) ||
          (jpfirst == 3 && jpsecond == 1)) {
        connect3d[i].ewhich_e1[m] = jpfirst - 1;
        connect3d[i].nside_e1[m] = OPPOSITE_SIDE;
        if (dotnorm < -1.0+flatthresh) connect3d[i].aflag_e1[m] = FLAT;
        else {
          MathExtra::cross3(inorm,jnorm,icrossj);
          if (MathExtra::dot3(icrossj,iedge) > 0.0)
            connect3d[i].aflag_e1[m] = CONCAVE;
          else
            connect3d[i].aflag_e1[m] = CONVEX;
        }
      } else {
        if (jpfirst == 2) connect3d[i].ewhich_e1[m] = 0;
        else if (jpfirst == 3) connect3d[i].ewhich_e1[m] = 1;
        else if (jpfirst == 1) connect3d[i].ewhich_e1[m] = 2;
        connect3d[i].nside_e1[m] = SAME_SIDE;
        if (dotnorm > 1.0-flatthresh) connect3d[i].aflag_e1[m] = FLAT;
        else {
          MathExtra::cross3(inorm,jnorm,icrossj);
          if (MathExtra::dot3(icrossj,iedge) < 0.0)
            connect3d[i].aflag_e1[m] = CONCAVE;
          else
            connect3d[i].aflag_e1[m] = CONVEX;
        }
      }
    }

    for (m = 0; m < connect3d[i].ne2; m++) {
      j = connect3d[i].neigh_e2[m];

      if (tris[i].p2 == tris[j].p1) jpfirst = 1;
      else if (tris[i].p2 == tris[j].p2) jpfirst = 2;
      else if (tris[i].p2 == tris[j].p3) jpfirst = 3;

      if (tris[i].p3 == tris[j].p1) jpsecond = 1;
      else if (tris[i].p3 == tris[j].p2) jpsecond = 2;
      else if (tris[i].p3 == tris[j].p3) jpsecond = 3;

      inorm = tris[i].norm;
      jnorm = tris[j].norm;
      dotnorm = MathExtra::dot3(inorm,jnorm);
      MathExtra::sub3(points[tris[i].p3].x,points[tris[i].p2].x,iedge);

      if ((jpfirst == 1 && jpsecond == 2) ||
          (jpfirst == 2 && jpsecond == 3) ||
          (jpfirst == 3 && jpsecond == 1)) {
        connect3d[i].ewhich_e2[m] = jpfirst - 1;
        connect3d[i].nside_e2[m] = OPPOSITE_SIDE;
        if (dotnorm < -1.0+flatthresh) connect3d[i].aflag_e2[m] = FLAT;
        else {
          MathExtra::cross3(inorm,jnorm,icrossj);
          if (MathExtra::dot3(icrossj,iedge) > 0.0)
            connect3d[i].aflag_e2[m] = CONCAVE;
          else
            connect3d[i].aflag_e2[m] = CONVEX;
        }
      } else {
        if (jpfirst == 2) connect3d[i].ewhich_e2[m] = 0;
        else if (jpfirst == 3) connect3d[i].ewhich_e2[m] = 1;
        else if (jpfirst == 1) connect3d[i].ewhich_e2[m] = 2;
        connect3d[i].nside_e2[m] = SAME_SIDE;
        if (dotnorm > 1.0-flatthresh) connect3d[i].aflag_e2[m] = FLAT;
        else {
          MathExtra::cross3(inorm,jnorm,icrossj);
          if (MathExtra::dot3(icrossj,iedge) < 0.0)
            connect3d[i].aflag_e2[m] = CONCAVE;
          else
            connect3d[i].aflag_e2[m] = CONVEX;
        }
      }
    }

    for (m = 0; m < connect3d[i].ne3; m++) {
      j = connect3d[i].neigh_e3[m];

      if (tris[i].p3 == tris[j].p1) jpfirst = 1;
      else if (tris[i].p3 == tris[j].p2) jpfirst = 2;
      else if (tris[i].p3 == tris[j].p3) jpfirst = 3;

      if (tris[i].p1 == tris[j].p1) jpsecond = 1;
      else if (tris[i].p1 == tris[j].p2) jpsecond = 2;
      else if (tris[i].p1 == tris[j].p3) jpsecond = 3;

      inorm = tris[i].norm;
      jnorm = tris[j].norm;
      dotnorm = MathExtra::dot3(inorm,jnorm);
      MathExtra::sub3(points[tris[i].p1].x,points[tris[i].p3].x,iedge);

      if ((jpfirst == 1 && jpsecond == 2) ||
          (jpfirst == 2 && jpsecond == 3) ||
          (jpfirst == 3 && jpsecond == 1)) {
        connect3d[i].ewhich_e3[m] = jpfirst - 1;
        connect3d[i].nside_e3[m] = OPPOSITE_SIDE;
        if (dotnorm < -1.0+flatthresh) connect3d[i].aflag_e3[m] = FLAT;
        else {
          MathExtra::cross3(inorm,jnorm,icrossj);
          if (MathExtra::dot3(icrossj,iedge) > 0.0)
            connect3d[i].aflag_e3[m] = CONCAVE;
          else
            connect3d[i].aflag_e3[m] = CONVEX;
        }
      } else {
        if (jpfirst == 2) connect3d[i].ewhich_e3[m] = 0;
        else if (jpfirst == 3) connect3d[i].ewhich_e3[m] = 1;
        else if (jpfirst == 1) connect3d[i].ewhich_e3[m] = 2;
        connect3d[i].nside_e3[m] = SAME_SIDE;
        if (dotnorm > 1.0-flatthresh) connect3d[i].aflag_e3[m] = FLAT;
        else {
          MathExtra::cross3(inorm,jnorm,icrossj);
          if (MathExtra::dot3(icrossj,iedge) < 0.0)
            connect3d[i].aflag_e3[m] = CONCAVE;
          else
            connect3d[i].aflag_e3[m] = CONVEX;
        }
      }
    }
  }

  // setup tri corner point connectivity lists
  // counts = # of tris containing each corner point (including self)
  // ctris = ragged 2d array with indices of tris which contain each point

  memory->create(counts,npoints,"surface/global:counts");

  for (int i = 0; i < npoints; i++) counts[i] = 0;

  for (int i = 0; i < ntris; i++) {
    counts[tris[i].p1]++;
    counts[tris[i].p2]++;
    counts[tris[i].p3]++;
  }

  int **ctris;
  memory->create_ragged(ctris,npoints,counts,"surface/global:ctris");

  for (int i = 0; i < npoints; i++) counts[i] = 0;

  for (int i = 0; i < ntris; i++) {
    ctris[tris[i].p1][counts[tris[i].p1]++] = i;
    ctris[tris[i].p2][counts[tris[i].p2]++] = i;
    ctris[tris[i].p3][counts[tris[i].p3]++] = i;
  }

  // c123_counts = # of tris connecting to corner points c123 of each tri
  // do NOT include self or tris which connect to an edge
  // only include tris which only connect at the corner point

  int *c1_counts,*c2_counts,*c3_counts;
  memory->create(c1_counts,ntris,"surface/global:c1_counts");
  memory->create(c2_counts,ntris,"surface/global:c2_counts");
  memory->create(c3_counts,ntris,"surface/global:c3_counts");

  for (int i = 0; i < ntris; i++) {
    c1_counts[i] = counts[tris[i].p1] - 1;
    c1_counts[i] -= connect3d[i].ne3 + connect3d[i].ne1;
    c2_counts[i] = counts[tris[i].p2] - 1;
    c2_counts[i] -= connect3d[i].ne1 + connect3d[i].ne2;
    c3_counts[i] = counts[tris[i].p3] - 1;
    c3_counts[i] -= connect3d[i].ne2 + connect3d[i].ne3;
  }

  // allocate all corner ragged arrays which Connect3d will point to

  memory->create_ragged(neigh_c1,ntris,c1_counts,"surface/global:neigh_c1");
  memory->create_ragged(neigh_c2,ntris,c2_counts,"surface/global:neigh_c2");
  memory->create_ragged(neigh_c3,ntris,c3_counts,"surface/global:neigh_c3");
  memory->create_ragged(cwhich_c1,ntris,c1_counts,"surface/global:cwhich_c1");
  memory->create_ragged(cwhich_c2,ntris,c2_counts,"surface/global:cwhich_c2");
  memory->create_ragged(cwhich_c3,ntris,c3_counts,"surface/global:cwhich_c3");

  // set connect3d corner vector ptrs to rows of corresponding corner ragged arrays

  for (int i = 0; i < ntris; i++) {
    connect3d[i].nc1 = c1_counts[i];
    if (connect3d[i].nc1) {
      connect3d[i].neigh_c1 = neigh_c1[i];
      connect3d[i].cwhich_c1 = cwhich_c1[i];
    } else {
      connect3d[i].neigh_c1 = nullptr;
      connect3d[i].cwhich_c1 = nullptr;
    }

    connect3d[i].nc2 = c2_counts[i];
    if (connect3d[i].nc2) {
      connect3d[i].neigh_c2 = neigh_c2[i];
      connect3d[i].cwhich_c2 = cwhich_c2[i];
    } else {
      connect3d[i].neigh_c2 = nullptr;
      connect3d[i].cwhich_c2 = nullptr;
    }

    connect3d[i].nc3 = c3_counts[i];
    if (connect3d[i].nc3) {
      connect3d[i].neigh_c3 = neigh_c3[i];
      connect3d[i].cwhich_c3 = cwhich_c3[i];
    } else {
      connect3d[i].neigh_c3 = nullptr;
      connect3d[i].cwhich_c3 = nullptr;
    }
  }

  // initialize connect3d corner neigh vectors for each corner of each tri
  // do NOT include self or tris which connect to an edge
  // only include tris which only connect at the corner point

  int n,medge,skipflag;

  for (int i = 0; i < ntris; i++) {
    if (connect3d[i].nc1) {
      j = 0;
      for (m = 0; m < counts[tris[i].p1]; m++) {
        n = ctris[tris[i].p1][m];
        if (n == i) continue;

        skipflag = 0;
        for (medge = 0; medge < connect3d[i].ne3; medge++)
          if (n == connect3d[i].neigh_e3[medge]) skipflag = 1;
        for (medge = 0; medge < connect3d[i].ne1; medge++)
          if (n == connect3d[i].neigh_e1[medge]) skipflag = 1;
        if (skipflag) continue;

        connect3d[i].neigh_c1[j] = n;
        j++;
      }
    }
    if (connect3d[i].nc2) {
      j = 0;
      for (m = 0; m < counts[tris[i].p2]; m++) {
        n = ctris[tris[i].p2][m];
        if (n == i) continue;

        skipflag = 0;
        for (medge = 0; medge < connect3d[i].ne1; medge++)
          if (n == connect3d[i].neigh_e1[medge]) skipflag = 1;
        for (medge = 0; medge < connect3d[i].ne2; medge++)
          if (n == connect3d[i].neigh_e2[medge]) skipflag = 1;
        if (skipflag) continue;

        connect3d[i].neigh_c2[j] = n;
        j++;
      }
    }
    if (connect3d[i].nc3) {
      j = 0;
      for (m = 0; m < counts[tris[i].p3]; m++) {
        n = ctris[tris[i].p3][m];
        if (n == i) continue;

        skipflag = 0;
        for (medge = 0; medge < connect3d[i].ne2; medge++)
          if (n == connect3d[i].neigh_e2[medge]) skipflag = 1;
        for (medge = 0; medge < connect3d[i].ne3; medge++)
          if (n == connect3d[i].neigh_e3[medge]) skipflag = 1;
        if (skipflag) continue;

        connect3d[i].neigh_c3[j] = n;
        j++;
      }
    }
  }

  // deallocate counts, ctris, c123_counts

  memory->destroy(counts);
  memory->destroy(ctris);
  memory->destroy(c1_counts);
  memory->destroy(c2_counts);
  memory->destroy(c3_counts);

  // set connect3d cwhich for each end point of each line
  // see fsg.h file for an explanation of each corner vector in Connect3d
  // aflag is based on dot and cross product of 2 connected tri normals

  for (int i = 0; i < ntris; i++) {
    for (m = 0; m < connect3d[i].nc1; m++) {
      j = connect3d[i].neigh_c1[m];
      if (tris[i].p1 == tris[j].p1) connect3d[i].cwhich_c1[m] = 0;
      else if (tris[i].p1 == tris[j].p2) connect3d[i].cwhich_c1[m] = 1;
      else if (tris[i].p1 == tris[j].p3) connect3d[i].cwhich_c1[m] = 2;
    }
    for (m = 0; m < connect3d[i].nc2; m++) {
      j = connect3d[i].neigh_c2[m];
      if (tris[i].p2 == tris[j].p1) connect3d[i].cwhich_c2[m] = 0;
      else if (tris[i].p2 == tris[j].p2) connect3d[i].cwhich_c2[m] = 1;
      else if (tris[i].p2 == tris[j].p3) connect3d[i].cwhich_c2[m] = 2;
    }
    for (m = 0; m < connect3d[i].nc3; m++) {
      j = connect3d[i].neigh_c3[m];
      if (tris[i].p3 == tris[j].p1) connect3d[i].cwhich_c3[m] = 0;
      else if (tris[i].p3 == tris[j].p2) connect3d[i].cwhich_c3[m] = 1;
      else if (tris[i].p3 == tris[j].p3) connect3d[i].cwhich_c3[m] = 2;
    }
  }
}

/* ----------------------------------------------------------------------
   print stats on all lines and their connections
------------------------------------------------------------------------- */

void FixSurfaceGlobal::stats2d()
{
  double size;
  double delta[3];

  int nconnect = 0;
  int nfree = 0;
  double minsize = BIG;
  double maxsize = 0.0;

  for (int i = 0; i < nlines; i++) {
    nconnect += connect2d[i].np1 + connect2d[i].np2;
    if (connect2d[i].np1 == 0) nfree++;
    if (connect2d[i].np2 == 0) nfree++;
    MathExtra::sub3(points[lines[i].p1].x,points[lines[i].p2].x,delta);
    size = MathExtra::len3(delta);
    minsize = MIN(minsize,size);
    maxsize = MAX(maxsize,size);
  }

  nconnect /= 2;

  if (comm->me == 0) {
    utils::logmesg(lmp,"Fix surface/global line segment creation:\n");
    utils::logmesg(lmp,fmt::format("  {} lines\n",nlines));
    utils::logmesg(lmp,fmt::format("  {} line end points\n",npoints));
    utils::logmesg(lmp,fmt::format("  {} end point connections\n",nconnect));
    utils::logmesg(lmp,fmt::format("  {} free end points\n",nfree));
    utils::logmesg(lmp,fmt::format("  {} min line length\n",minsize));
    utils::logmesg(lmp,fmt::format("  {} max line length\n",maxsize));
  }
}

/* ----------------------------------------------------------------------
   print stats on all tris and their connections
------------------------------------------------------------------------- */

void FixSurfaceGlobal::stats3d()
{
  double size,area;
  double delta[3],edge12[3],edge13[3],cross[3];

  int nconnect_edge = 0;
  int nconnect_corner = 0;
  int nfree_edge = 0;
  int nfree_corner = 0;
  double minedge = BIG;
  double maxedge = 0.0;
  double minarea = BIG;
  double maxarea = 0.0;

  for (int i = 0; i < ntris; i++) {
    nconnect_edge += connect3d[i].ne1 + connect3d[i].ne2 + connect3d[i].ne3;
    nconnect_corner += connect3d[i].nc1 + connect3d[i].nc2 + connect3d[i].nc3;

    if (connect3d[i].ne1 == 0) nfree_edge++;
    if (connect3d[i].ne2 == 0) nfree_edge++;
    if (connect3d[i].ne3 == 0) nfree_edge++;

    // a free corner point requires 2 adjacent edges also have no connections

    if (connect3d[i].nc1 == 0 && (connect3d[i].ne3 == 0 && connect3d[i].ne1 == 0)) nfree_corner++;
    if (connect3d[i].nc2 == 0 && (connect3d[i].ne1 == 0 && connect3d[i].ne2 == 0)) nfree_corner++;
    if (connect3d[i].nc3 == 0 && (connect3d[i].ne2 == 0 && connect3d[i].ne3 == 0)) nfree_corner++;

    MathExtra::sub3(points[tris[i].p1].x,points[tris[i].p2].x,delta);
    size = MathExtra::len3(delta);
    minedge = MIN(minedge,size);
    maxedge = MAX(maxedge,size);
    MathExtra::sub3(points[tris[i].p2].x,points[tris[i].p3].x,delta);
    size = MathExtra::len3(delta);
    minedge = MIN(minedge,size);
    maxedge = MAX(maxedge,size);
    MathExtra::sub3(points[tris[i].p3].x,points[tris[i].p1].x,delta);
    size = MathExtra::len3(delta);
    minedge = MIN(minedge,size);
    maxedge = MAX(maxedge,size);

    MathExtra::sub3(points[tris[i].p2].x,points[tris[i].p1].x,edge12);
    MathExtra::sub3(points[tris[i].p3].x,points[tris[i].p1].x,edge13);
    MathExtra::cross3(edge12,edge13,cross);
    area = 0.5 * MathExtra::len3(cross);
    minarea = MIN(minarea,area);
    maxarea = MAX(maxarea,area);
  }

  nconnect_edge /= 2;
  nconnect_corner /= 2;

  if (comm->me == 0) {
    utils::logmesg(lmp,"Fix surface/global triangle creation:\n");
    utils::logmesg(lmp,fmt::format("  {} tris\n",ntris));
    utils::logmesg(lmp,fmt::format("  {} tri edges\n",nedges));
    utils::logmesg(lmp,fmt::format("  {} tri corner points\n",npoints));
    utils::logmesg(lmp,fmt::format("  {} edge connections\n",nconnect_edge));
    utils::logmesg(lmp,fmt::format("  {} corner point connections\n",nconnect_corner));
    utils::logmesg(lmp,fmt::format("  {} free edges\n",nfree_edge));
    utils::logmesg(lmp,fmt::format("  {} free corner points\n",nfree_corner));
    utils::logmesg(lmp,fmt::format("  {} min edge length\n",minedge));
    utils::logmesg(lmp,fmt::format("  {} max edge length\n",maxedge));
    utils::logmesg(lmp,fmt::format("  {} min tri area\n",minarea));
    utils::logmesg(lmp,fmt::format("  {} max tri area\n",maxarea));
  }
}

/* ----------------------------------------------------------------------
   set attributes of all lines or tris
   xsurf,vsurf,omegasurf,norm
------------------------------------------------------------------------- */

void FixSurfaceGlobal::surface_attributes()
{
  double delta[3],p12[3],p13[3];
  double *p1,*p2,*p3;
  double zunit[3] = {0.0,0.0,1.0};

  memory->create(xsurf,nsurf,3,"surface/global:xsurf");
  memory->create(vsurf,nsurf,3,"surface/global:vsurf");
  memory->create(omegasurf,nsurf,3,"surface/global:omegasurf");
  memory->create(radsurf,nsurf,"surface/global:radsurf");

  if (dimension == 2) {
    for (int i = 0; i < nsurf; i++) {
      p1 = points[lines[i].p1].x;
      p2 = points[lines[i].p2].x;
      xsurf[i][0] = 0.5 * (p1[0]+p2[0]);
      xsurf[i][1] = 0.5 * (p1[1]+p2[1]);
      xsurf[i][2] = 0.0;

      MathExtra::sub3(p2,p1,p12);
      radsurf[i] = 0.5 * MathExtra::len3(p12);

      MathExtra::cross3(zunit,p12,lines[i].norm);
      MathExtra::norm3(lines[i].norm);
    }

  } else {
    for (int i = 0; i < nsurf; i++) {
      p1 = points[tris[i].p1].x;
      p2 = points[tris[i].p2].x;
      p3 = points[tris[i].p3].x;
      xsurf[i][0] = (p1[0]+p2[0]+p3[0]) / 3.0;
      xsurf[i][1] = (p1[1]+p2[1]+p3[1]) / 3.0;
      xsurf[i][2] = (p1[2]+p2[2]+p3[2]) / 3.0;

      MathExtra::sub3(p1,xsurf[i],delta);
      radsurf[i] = MathExtra::lensq3(delta);
      MathExtra::sub3(p2,xsurf[i],delta);
      radsurf[i] = MAX(radsurf[i],MathExtra::lensq3(delta));
      MathExtra::sub3(p3,xsurf[i],delta);
      radsurf[i] = MAX(radsurf[i],MathExtra::lensq3(delta));
      radsurf[i] = sqrt(radsurf[i]);

      MathExtra::sub3(p2,p1,p12);
      MathExtra::sub3(p3,p1,p13);
      MathExtra::cross3(p12,p13,tris[i].norm);
      MathExtra::norm3(tris[i].norm);
    }
  }

  for (int i = 0; i < nsurf; i++) {
    vsurf[i][0] = vsurf[i][1] = vsurf[i][2] = 0.0;
    omegasurf[i][0] = omegasurf[i][1] = omegasurf[i][2] = 0.0;
  }
}

/* -------------------------------------------------------------------------
   linear move: X = X0 + V*dt
------------------------------------------------------------------------- */

void FixSurfaceGlobal::move_linear(int imotion, int i)
{
  Motion *motion = &motions[imotion];
  double time_origin = motion->time_origin;
  double delta = (update->ntimestep - time_origin) * dt;

  int vxflag = motion->vxflag;
  int vyflag = motion->vyflag;
  int vzflag = motion->vzflag;
  double vx = motion->vx;
  double vy = motion->vy;
  double vz = motion->vz;

  // points - use of pointmove only moves a point once

  int pindex;
  double *pt;

  if (dimension == 2) {
    pindex = lines[i].p1;
    if (!pointmove[pindex]) {
      pt = points[pindex].x;
      if (vxflag) pt[0] = points_original[i][0] + vx * delta;
      if (vyflag) pt[1] = points_original[i][1] + vy * delta;
      pointmove[pindex] = 1;
    }
    pindex = lines[i].p2;
    if (!pointmove[pindex]) {
      pt = points[pindex].x;
      if (vxflag) pt[0] = points_original[i][0] + vx * delta;
      if (vyflag) pt[1] = points_original[i][1] + vy * delta;
      pointmove[pindex] = 1;
    }

  } else {
    pindex = tris[i].p1;
    if (!pointmove[pindex]) {
      pt = points[pindex].x;
      if (vxflag) pt[0] = points_original[i][0] + vx * delta;
      if (vyflag) pt[1] = points_original[i][1] + vy * delta;
      if (vzflag) pt[2] = points_original[i][2] + vz * delta;
      pointmove[pindex] = 1;
    }
    pindex = tris[i].p2;
    if (!pointmove[pindex]) {
      pt = points[pindex].x;
      if (vxflag) pt[0] = points_original[i][0] + vx * delta;
      if (vyflag) pt[1] = points_original[i][1] + vy * delta;
      if (vzflag) pt[2] = points_original[i][2] + vz * delta;
      pointmove[pindex] = 1;
    }
    pindex = tris[i].p3;
    if (!pointmove[pindex]) {
      pt = points[pindex].x;
      if (vxflag) pt[0] = points_original[i][0] + vx * delta;
      if (vyflag) pt[1] = points_original[i][1] + vy * delta;
      if (vzflag) pt[2] = points_original[i][2] + vz * delta;
      pointmove[pindex] = 1;
    }
  }

  // xsurf and vsurf

  if (vxflag) {
    vsurf[i][0] = vx;
    xsurf[i][0] = xsurf_original[i][0] + vx * delta;
  }
  if (vyflag) {
    vsurf[i][1] = vy;
    xsurf[i][1] = xsurf_original[i][1] + vy * delta;
  }
  if (vzflag) {
    vsurf[i][2] = vz;
    xsurf[i][2] = xsurf_original[i][2] + vz * delta;
  }
}

/* -------------------------------------------------------------------------
   wiggle move: X = X0 + A sin(w*dt)
------------------------------------------------------------------------- */

void FixSurfaceGlobal::move_wiggle(int imotion, int i)
{
  Motion *motion = &motions[imotion];
  double time_origin = motion->time_origin;
  double omega = motion->omega;
  double delta = (update->ntimestep - time_origin) * dt;
  double arg = omega * delta;
  double sine = sin(arg);
  double cosine = cos(arg);

  int axflag = motion->axflag;
  int ayflag = motion->ayflag;
  int azflag = motion->azflag;
  double ax = motion->ax;
  double ay = motion->ay;
  double az = motion->az;

  // points - use of pointmove only moves a point once

  int pindex;
  double *pt;

  if (dimension == 2) {
    pindex = lines[i].p1;
    if (!pointmove[pindex]) {
      pt = points[pindex].x;
      if (axflag) pt[0] = points_original[i][0] + ax * sine;
      if (ayflag) pt[1] = points_original[i][1] + ay * sine;
      pointmove[pindex] = 1;
    }
    pindex = lines[i].p2;
    if (!pointmove[pindex]) {
      pt = points[pindex].x;
      if (axflag) pt[0] = points_original[i][0] + ax * sine;
      if (ayflag) pt[1] = points_original[i][1] + ay * sine;
      pointmove[pindex] = 1;
    }

  } else {
    pindex = tris[i].p1;
    if (!pointmove[pindex]) {
      pt = points[pindex].x;
      if (axflag) pt[0] = points_original[i][0] + ax * sine;
      if (ayflag) pt[1] = points_original[i][1] + ay * sine;
      if (azflag) pt[2] = points_original[i][2] + az * sine;
      pointmove[pindex] = 1;
    }
    pindex = tris[i].p2;
    if (!pointmove[pindex]) {
      pt = points[pindex].x;
      if (axflag) pt[0] = points_original[i][0] + ax * sine;
      if (ayflag) pt[1] = points_original[i][1] + ay * sine;
      if (azflag) pt[2] = points_original[i][2] + az * sine;
      pointmove[pindex] = 1;
    }
    pindex = tris[i].p3;
    if (!pointmove[pindex]) {
      pt = points[pindex].x;
      if (axflag) pt[0] = points_original[i][0] + ax * sine;
      if (ayflag) pt[1] = points_original[i][1] + ay * sine;
      if (azflag) pt[2] = points_original[i][2] + az * sine;
      pointmove[pindex] = 1;
    }
  }

  // xsurf and vsurf

  if (axflag) {
    vsurf[i][0] = ax * omega * cosine;
    xsurf[i][0] = xsurf_original[i][0] + ax * sine;
  }
  if (ayflag) {
    vsurf[i][1] = ay * omega * cosine;
    xsurf[i][1] = xsurf_original[i][1] + ay * sine;
  }
  if (azflag) {
    vsurf[i][2] = az * omega * cosine;
    xsurf[i][2] = xsurf_original[i][2] + az * sine;
  }
}

/* -------------------------------------------------------------------------
   rotate move: rotate by right-hand rule around omega
------------------------------------------------------------------------- */

void FixSurfaceGlobal::move_rotate(int imotion, int i)
{
  Motion *motion = &motions[imotion];

  double time_origin = motion->time_origin;
  double omega = motion->omega;
  double *rpoint = motion->point;
  double *runit = motion->unit;

  double delta = (update->ntimestep - time_origin) * dt;
  double arg = omega * delta;
  double cosine = cos(arg);
  double sine = sin(arg);

  // P = point = vector = point of rotation
  // R = vector = axis of rotation
  // w = omega of rotation (from period)
  // X0 = xoriginal = initial coord of atom
  // R0 = runit = unit vector for R
  // D = X0 - P = vector from P to X0
  // C = (D dot R0) R0 = projection of atom coord onto R line
  // A = D - C = vector from R line to X0
  // B = R0 cross A = vector perp to A in plane of rotation
  // A,B define plane of circular rotation around R line
  // X = P + C + A cos(w*dt) + B sin(w*dt)
  // V = w R0 cross (A cos(w*dt) + B sin(w*dt))

  // points - use of pointmove only moves a point once

  int pindex;

  if (dimension == 2) {
    pindex = lines[i].p1;
    if (!pointmove[pindex]) {
      move_rotate_point(pindex,rpoint,runit,cosine,sine);
      pointmove[pindex] = 1;
    }
    pindex = lines[i].p2;
    if (!pointmove[pindex]) {
      move_rotate_point(pindex,rpoint,runit,cosine,sine);
      pointmove[pindex] = 1;
    }

  } else {
    pindex = tris[i].p1;
    if (!pointmove[pindex]) {
      move_rotate_point(pindex,rpoint,runit,cosine,sine);
      pointmove[pindex] = 1;
    }
    pindex = tris[i].p2;
    if (!pointmove[pindex]) {
      move_rotate_point(pindex,rpoint,runit,cosine,sine);
      pointmove[pindex] = 1;
    }
    pindex = tris[i].p3;
    if (!pointmove[pindex]) {
      move_rotate_point(pindex,rpoint,runit,cosine,sine);
      pointmove[pindex] = 1;
    }
  }

  // xsurf and vsurf

  double ddotr;
  double a[3],b[3],c[3],d[3],disp[3];

  d[0] = xsurf_original[i][0] - rpoint[0];
  d[1] = xsurf_original[i][1] - rpoint[1];
  d[2] = xsurf_original[i][2] - rpoint[2];
  ddotr = d[0]*runit[0] + d[1]*runit[1] + d[2]*runit[2];
  c[0] = ddotr*runit[0];
  c[1] = ddotr*runit[1];
  c[2] = ddotr*runit[2];
  a[0] = d[0] - c[0];
  a[1] = d[1] - c[1];
  a[2] = d[2] - c[2];
  b[0] = runit[1]*a[2] - runit[2]*a[1];
  b[1] = runit[2]*a[0] - runit[0]*a[2];
  b[2] = runit[0]*a[1] - runit[1]*a[0];
  disp[0] = a[0]*cosine  + b[0]*sine;
  disp[1] = a[1]*cosine  + b[1]*sine;
  disp[2] = a[2]*cosine  + b[2]*sine;

  xsurf[i][0] = rpoint[0] + c[0] + disp[0];
  xsurf[i][1] = rpoint[1] + c[1] + disp[1];
  xsurf[i][2] = rpoint[2] + c[2] + disp[2];
  vsurf[i][0] = omega * (runit[1]*disp[2] - runit[2]*disp[1]);
  vsurf[i][1] = omega * (runit[2]*disp[0] - runit[0]*disp[2]);
  vsurf[i][2] = omega * (runit[0]*disp[1] - runit[1]*disp[0]);

  // normals

  double p12[3],p13[3];
  double *p1,*p2,*p3;

  if (dimension == 2) {
    double zunit[3] = {0.0,0.0,1.0};
    p1 = points[lines[i].p1].x;
    p2 = points[lines[i].p2].x;
    MathExtra::sub3(p2,p1,p12);
    MathExtra::cross3(zunit,p12,lines[i].norm);
    MathExtra::norm3(lines[i].norm);

  } else {
    p1 = points[tris[i].p1].x;
    p2 = points[tris[i].p2].x;
    p3 = points[tris[i].p3].x;
    MathExtra::sub3(p1,p2,p12);
    MathExtra::sub3(p1,p3,p13);
    MathExtra::cross3(p12,p13,tris[i].norm);
    MathExtra::norm3(tris[i].norm);
  }
}

/* -------------------------------------------------------------------------
   transrotate move:
   rotate by right-hand rule around omega
   add translation after rotation
------------------------------------------------------------------------- */

void FixSurfaceGlobal::move_transrotate(int imotion, int i)
{
  Motion *motion = &motions[imotion];

  double time_origin = motion->time_origin;
  double omega = motion->omega;
  double *rpoint = motion->point;
  double *runit = motion->unit;

  double delta = (update->ntimestep - time_origin) * dt;
  double arg = omega * delta;
  double cosine = cos(arg);
  double sine = sin(arg);

  int vxflag = motion->vxflag;
  int vyflag = motion->vyflag;
  int vzflag = motion->vzflag;
  double vx = motion->vx;
  double vy = motion->vy;
  double vz = motion->vz;

  // P = point = vector = point of rotation
  // R = vector = axis of rotation
  // w = omega of rotation (from period)
  // X0 = xoriginal = initial coord of atom
  // R0 = runit = unit vector for R
  // D = X0 - P = vector from P to X0
  // C = (D dot R0) R0 = projection of atom coord onto R line
  // A = D - C = vector from R line to X0
  // B = R0 cross A = vector perp to A in plane of rotation
  // A,B define plane of circular rotation around R line
  // X = P + C + A cos(w*dt) + B sin(w*dt)
  // V = w R0 cross (A cos(w*dt) + B sin(w*dt))

  // points - use of pointmove only moves a point once

  int pindex;
  double *pt;

  if (dimension == 2) {
    pindex = lines[i].p1;
    pt = points[pindex].x;
    if (!pointmove[pindex]) {
      move_rotate_point(pindex,rpoint,runit,cosine,sine);
      if (vxflag) pt[0] += vx * delta;
      if (vyflag) pt[1] += vy * delta;
      pointmove[pindex] = 1;
    }
    pindex = lines[i].p2;
    pt = points[pindex].x;
    if (!pointmove[pindex]) {
      move_rotate_point(pindex,rpoint,runit,cosine,sine);
      if (vxflag) pt[0] += vx * delta;
      if (vyflag) pt[1] += vy * delta;
      pointmove[pindex] = 1;
    }

  } else {
    pindex = tris[i].p1;
    pt = points[pindex].x;
    if (!pointmove[pindex]) {
      move_rotate_point(pindex,rpoint,runit,cosine,sine);
      if (vxflag) pt[0] += vx * delta;
      if (vyflag) pt[1] += vy * delta;
      if (vzflag) pt[2] += vz * delta;
      pointmove[pindex] = 1;
    }
    pindex = tris[i].p2;
    pt = points[pindex].x;
    if (!pointmove[pindex]) {
      move_rotate_point(pindex,rpoint,runit,cosine,sine);
      if (vxflag) pt[0] += vx * delta;
      if (vyflag) pt[1] += vy * delta;
      if (vzflag) pt[2] += vz * delta;
      pointmove[pindex] = 1;
    }
    pindex = tris[i].p3;
    pt = points[pindex].x;
    if (!pointmove[pindex]) {
      move_rotate_point(pindex,rpoint,runit,cosine,sine);
      if (vxflag) pt[0] += vx * delta;
      if (vyflag) pt[1] += vy * delta;
      if (vzflag) pt[2] += vz * delta;
      pointmove[pindex] = 1;
    }
  }

  // xsurf and vsurf

  double ddotr;
  double a[3],b[3],c[3],d[3],disp[3];

  d[0] = xsurf_original[i][0] - rpoint[0];
  d[1] = xsurf_original[i][1] - rpoint[1];
  d[2] = xsurf_original[i][2] - rpoint[2];
  ddotr = d[0]*runit[0] + d[1]*runit[1] + d[2]*runit[2];
  c[0] = ddotr*runit[0];
  c[1] = ddotr*runit[1];
  c[2] = ddotr*runit[2];
  a[0] = d[0] - c[0];
  a[1] = d[1] - c[1];
  a[2] = d[2] - c[2];
  b[0] = runit[1]*a[2] - runit[2]*a[1];
  b[1] = runit[2]*a[0] - runit[0]*a[2];
  b[2] = runit[0]*a[1] - runit[1]*a[0];
  disp[0] = a[0]*cosine  + b[0]*sine;
  disp[1] = a[1]*cosine  + b[1]*sine;
  disp[2] = a[2]*cosine  + b[2]*sine;

  xsurf[i][0] = rpoint[0] + c[0] + disp[0];
  xsurf[i][1] = rpoint[1] + c[1] + disp[1];
  xsurf[i][2] = rpoint[2] + c[2] + disp[2];
  vsurf[i][0] = omega * (runit[1]*disp[2] - runit[2]*disp[1]); + vxflag*vx;
  vsurf[i][1] = omega * (runit[2]*disp[0] - runit[0]*disp[2]); + vyflag*vy;
  vsurf[i][2] = omega * (runit[0]*disp[1] - runit[1]*disp[0]);

  if (vxflag) xsurf[i][0] += vx*delta;
  if (vyflag) xsurf[i][1] += vy*delta;
  if (vzflag) xsurf[i][2] += vz*delta;
  if (vxflag) vsurf[i][0] += vx;
  if (vyflag) vsurf[i][1] += vy;
  if (vzflag) vsurf[i][2] += vz;

  // normals

  double p12[3],p13[3];
  double *p1,*p2,*p3;

  if (dimension == 2) {
    double zunit[3] = {0.0,0.0,1.0};
    p1 = points[lines[i].p1].x;
    p2 = points[lines[i].p2].x;
    MathExtra::sub3(p2,p1,p12);
    MathExtra::cross3(zunit,p12,lines[i].norm);
    MathExtra::norm3(lines[i].norm);

  } else {
    p1 = points[tris[i].p1].x;
    p2 = points[tris[i].p2].x;
    p3 = points[tris[i].p3].x;
    MathExtra::sub3(p1,p2,p12);
    MathExtra::sub3(p1,p3,p13);
    MathExtra::cross3(p12,p13,tris[i].norm);
    MathExtra::norm3(tris[i].norm);
  }
}

/* -------------------------------------------------------------------------
   rotate point I by right-hand rule around omega
/* ------------------------------------------------------------------------- */

void FixSurfaceGlobal::move_rotate_point(int i, double *rpoint, double *runit,
                                         double cosine, double sine)
{
  double a[3],b[3],c[3],d[3],disp[3];

  d[0] = points_original[i][0] - rpoint[0];
  d[1] = points_original[i][1] - rpoint[1];
  d[2] = points_original[i][2] - rpoint[2];

  double ddotr = d[0]*runit[0] + d[1]*runit[1] + d[2]*runit[2];
  c[0] = ddotr*runit[0];
  c[1] = ddotr*runit[1];
  c[2] = ddotr*runit[2];
  a[0] = d[0] - c[0];
  a[1] = d[1] - c[1];
  a[2] = d[2] - c[2];
  b[0] = runit[1]*a[2] - runit[2]*a[1];
  b[1] = runit[2]*a[0] - runit[0]*a[2];
  b[2] = runit[0]*a[1] - runit[1]*a[0];
  disp[0] = a[0]*cosine  + b[0]*sine;
  disp[1] = a[1]*cosine  + b[1]*sine;
  disp[2] = a[2]*cosine  + b[2]*sine;

  double *pt = points[i].x;
  pt[0] = rpoint[0] + c[0] + disp[0];
  pt[1] = rpoint[1] + c[1] + disp[1];
  pt[2] = rpoint[2] + c[2] + disp[2];
}

/* -------------------------------------------------------------------------
   variable move:
   apply displacement variables if not NULL
   apply velocity variables if not NULL
------------------------------------------------------------------------- */

void FixSurfaceGlobal::move_variable(int imotion, int i)
{
  Motion *motion = &motions[imotion];

  char *xvarstr = motion->xvarstr;
  char *yvarstr = motion->yvarstr;
  char *zvarstr = motion->zvarstr;
  char *vxvarstr = motion->vxvarstr;
  char *vyvarstr = motion->vyvarstr;
  char *vzvarstr = motion->vzvarstr;
  double dx = motion->dx;
  double dy = motion->dy;
  double dz = motion->dz;
  double vx = motion->vx;
  double vy = motion->vy;
  double vz = motion->vz;

  double dt = update->dt;

  // points - use of pointmove only moves a point once
  // if displacement is variable, set x
  // if only velocity is variable, time-integrate x using v
  // if neither is variable, no change to x

  int pindex;
  double *pt;

  if (dimension == 2) {
    pindex = lines[i].p1;
    if (!pointmove[pindex]) {
      pt = points[pindex].x;
      if (vxvarstr && !xvarstr) pt[0] += dt*dx;
      else if (xvarstr) pt[0] = points_original[i][0] + dx;
      if (vyvarstr && !yvarstr) pt[1] += dt*dy;
      else if (yvarstr) pt[1] = points_original[i][1] + dy;
      pointmove[pindex] = 1;
    }
    pindex = lines[i].p2;
    if (!pointmove[pindex]) {
      pt = points[pindex].x;
      if (vxvarstr && !xvarstr) pt[0] += dt*dx;
      else if (xvarstr) pt[0] = points_original[i][0] + dx;
      if (vyvarstr && !yvarstr) pt[1] += dt*dy;
      else if (yvarstr) pt[1] = points_original[i][1] + dy;
      pointmove[pindex] = 1;
    }

  } else {
    pindex = tris[i].p1;
    if (!pointmove[pindex]) {
      pt = points[pindex].x;
      if (vxvarstr && !xvarstr) pt[0] += dt*dx;
      else if (xvarstr) pt[0] = points_original[i][0] + dx;
      if (vyvarstr && !yvarstr) pt[1] += dt*dy;
      else if (yvarstr) pt[1] = points_original[i][1] + dy;
      if (vzvarstr && !zvarstr) pt[2] += dt*dz;
      else if (zvarstr) pt[2] = points_original[i][2] + dz;
      pointmove[pindex] = 1;
    }
    pindex = tris[i].p2;
    if (!pointmove[pindex]) {
      pt = points[pindex].x;
      if (vxvarstr && !xvarstr) pt[0] += dt*dx;
      else if (xvarstr) pt[0] = points_original[i][0] + dx;
      if (vyvarstr && !yvarstr) pt[1] += dt*dy;
      else if (yvarstr) pt[1] = points_original[i][1] + dy;
      if (vzvarstr && !zvarstr) pt[2] += dt*dz;
      else if (zvarstr) pt[2] = points_original[i][2] + dz;
      pointmove[pindex] = 1;
    }
    pindex = tris[i].p3;
    if (!pointmove[pindex]) {
      pt = points[pindex].x;
      if (vxvarstr && !xvarstr) pt[0] += dt*dx;
      else if (xvarstr) pt[0] = points_original[i][0] + dx;
      if (vyvarstr && !yvarstr) pt[1] += dt*dy;
      else if (yvarstr) pt[1] = points_original[i][1] + dy;
      if (vzvarstr && !zvarstr) pt[2] += dt*dz;
      else if (zvarstr) pt[2] = points_original[i][2] + dz;
      pointmove[pindex] = 1;
    }
  }

  // xsurf and vsurf
  // if both displacement and velocity are variable, use them to set both x and v
  // if only displacement is variable, only set x
  // if only velocity is variable, set v and time-integrate x using v
  // if neither is variable, no change to x and v

  if (xvarstr && vxvarstr) {
    vsurf[i][0] = vx;
    xsurf[i][0] = xsurf_original[i][0] + dx;
  } else if (xvarstr) {
    xsurf[i][0] = xsurf_original[i][0] + dx;
  } else if (vxvarstr) {
    vsurf[i][0] = vx;
    xsurf[i][0] += dt*dx;
  }

  if (yvarstr && vyvarstr) {
    vsurf[i][1] = vy;
    xsurf[i][1] = xsurf_original[i][1] + dy;
  } else if (yvarstr) {
    xsurf[i][1] = xsurf_original[i][1] + dy;
  } else if (vyvarstr) {
    vsurf[i][1] = vy;
    xsurf[i][1] += dt*dy;
  }

  if (zvarstr && vzvarstr) {
    vsurf[i][2] = vx;
    xsurf[i][2] = xsurf_original[i][2] + dz;
  } else if (zvarstr) {
    xsurf[i][2] = xsurf_original[i][2] + dz;
  } else if (vzvarstr) {
    vsurf[i][2] = vz;
    xsurf[i][2] += dt*dz;
  }
}

/* ----------------------------------------------------------------------
   recursively through flat connections and process any contacts
------------------------------------------------------------------------- */

void FixSurfaceGlobal::walk_flat_connections2d(int j, int nsidej, std::vector<int> *flat_surfs, std::unordered_set<int> *processed_contacts, std::unordered_set<int> *hidden_contacts, std::map<int, int> *contacts_map)
{
  processed_contacts->insert(j);
  flat_surfs->push_back((*contacts_map)[j]);

  int n = (*contacts_map)[j];
  int k, m, aflag, nsidek, pwhich, shared_pt_j, shared_pt_k, nconnect;
  int convex_flag;
  for (nconnect = 0; nconnect < (connect2d[j].np1 + connect2d[j].np2); nconnect++) {
    shared_pt_j = 0;
    if (nconnect < connect2d[j].np1) {
      k = connect2d[j].neigh_p1[nconnect];
      aflag = connect2d[j].aflag_p1[nconnect];
      nsidek = connect2d[j].nside_p1[nconnect];
      pwhich = connect2d[j].pwhich_p1[nconnect];
      if (contact_surfs[n].jflag == -1)
        shared_pt_j = 1;
      if (pwhich == 0 && contact_surfs[n].jflag == -1)
        shared_pt_j = 1;
    } else {
      k = connect2d[j].neigh_p2[nconnect - connect2d[j].np1];
      aflag = connect2d[j].aflag_p2[nconnect - connect2d[j].np1];
      nsidek = connect2d[j].nside_p2[nconnect - connect2d[j].np1];
      pwhich = connect2d[j].pwhich_p2[nconnect - connect2d[j].np1];
      if (contact_surfs[n].jflag == -2)
        shared_pt_j = 1;
    }

    // Skip if processed
    if (processed_contacts->find(k) != processed_contacts->end())
      continue;

    // Skip if not in contact
    if (contacts_map->find(k) == contacts_map->end())
      continue;
    m = (*contacts_map)[k];
    shared_pt_k = 0;
    if ((pwhich == 0 && contact_surfs[m].jflag == -1) ||
      (pwhich == 1 && contact_surfs[m].jflag == -2))
      shared_pt_k = 1;

    // If connection is concave, ignore point
    convex_flag = 0;
    if ((nsidej == SAME_SIDE && aflag == CONVEX) ||
      (nsidej == OPPOSITE_SIDE && aflag == CONCAVE))
      convex_flag = 1;


    if (convex_flag) {
      hidden_contacts->insert(k);
    } else {
      if (shared_pt_j) {
        contact_surfs[n].ignore_pt = 1;
      } if (shared_pt_k) {
        contact_surfs[m].ignore_pt = 1;
      }
    }

    // Walk if flat, otherwise process later
    if (aflag != FLAT) continue;

    // store which side is connected to surf j's nside
    if (nsidej == OPPOSITE_SIDE) {
      if (nsidek == OPPOSITE_SIDE)
        nsidek = SAME_SIDE;
      else
        nsidek = OPPOSITE_SIDE;
    }
    contact_surfs[m].nside = nsidek;

    // check whether to skip end point
    shared_pt_k = 0;
    if ((pwhich == 0 && contact_surfs[m].jflag == -1) || (pwhich == 1 && contact_surfs[m].jflag == -2))
      shared_pt_k = 1;

    if (shared_pt_j)
      contact_surfs[n].ignore_pt = 1;
    if (shared_pt_k)
      contact_surfs[m].ignore_pt = 1;

    walk_flat_connections2d(k, nsidek, flat_surfs, processed_contacts, hidden_contacts, contacts_map);
  }
}
