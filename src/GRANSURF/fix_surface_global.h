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

#ifdef FIX_CLASS
// clang-format off
FixStyle(surface/global,FixSurfaceGlobal)
// clang-format on
#else

#ifndef LMP_FIX_SURFACE_GLOBAL_H
#define LMP_FIX_SURFACE_GLOBAL_H

#include <stdio.h>
#include "fix.h"
#include <map>
#include <unordered_set>
#include <tuple>
#include <vector>

namespace LAMMPS_NS {

namespace Granular_NS {
  class GranularModel;
}

class FixSurfaceGlobal : public Fix {
 public:

  // neighbor lists for spheres with surfs and shear history
  // accessed by fix shear/history

  class NeighList *list;
  class NeighList *listhistory;

  FixSurfaceGlobal(class LAMMPS *, int, char **);
  ~FixSurfaceGlobal();
  int setmask() override;
  void post_constructor() override;
  void init() override;
  void setup_pre_neighbor() override;
  void initial_integrate(int) override;
  void pre_neighbor() override;
  void post_force(int) override;

  int modify_param(int, char **x) override;
  void reset_dt() override;
  double memory_usage() override;

  void *extract(const char *, int &) override;
  int image(int *&, double **&) override;

 private:
  int dimension,firsttime,use_history;
  double dt,skin;
  double flatthresh;
  double Twall;
  int tvar;
  char *tstr;

  // per-surf properties

  int maxsurftype;
  double **xsurf,**vsurf,**omegasurf,*radsurf;

  // granular models

  struct ModelTypes {
    int plo,phi;
    int slo,shi;
  };

  ModelTypes *modeltypes;
  class Granular_NS::GranularModel **models;   // list of command-line models
  class Granular_NS::GranularModel ***types2model;  // model assigned to each particle/surf type pair

  int nmodel, maxmodel;
  int history, size_history, heat_flag;

  // neighbor params

  double triggersq;

  // settings for motion applied to specific surf types

  struct Motion {
    int active;
    int mstyle;
    int vxflag,vyflag,vzflag;
    int axflag,ayflag,azflag;
    int xvar,yvar,zvar;
    int vxvar,vyvar,vzvar;
    double vx,vy,vz;
    double ax,ay,az;
    double dx,dy,dz;
    double period;
    double point[3],axis[3],unit[3];
    double omega;
    char *xvarstr,*yvarstr,*zvarstr;
    char *vxvarstr,*vyvarstr,*vzvarstr;
    double time_origin;
  };

  struct Motion *motions;  // list of defined motions, can be flagged inactive
  int nmotion,maxmotion;   // # of defined motions versus allocated size
  int anymove;             // 1 if any surf motion is enabled
  int anymove_variable;    // 1 if any surf motion is style VARIABLE

  int *type2motion;        // assingment of surf types (1 to Ntype) to motions
                           // -1 = non-moving surf type

  double **points_original,**xsurf_original;
  double **points_lastneigh;
  int *pointmove;

  // storage of granular history info

  class FixNeighHistory *fix_history;
  double *zeroes;

  // rigid body masses for use in granular interactions

  class Fix *fix_rigid;    // ptr to rigid body fix, NULL if none
  double *mass_rigid;      // rigid mass for owned+ghost atoms
  int nmax;                // allocated size of mass_rigid

  // data structs for extracting surfs from molecule or STL files

  struct Point {
    double x[3];
  };

  struct Line {
    int mol,type;           // molID and type of the line
    int p1,p2;              // indices of points in line segment
    double norm[3];         // unit normal to line = Z x (p2-p1)
  };

  struct Tri {
    int mol,type;           // modID and type of the triangle
    int p1,p2,p3;           // indices of points in triangle
    double norm[3];         // unit normal to tri plane = (p2-p1) x (p3-p1)
  };

  Point *points;              // global list of unique points
  Line *lines;                // global list of lines
  Tri *tris;                  // global list of tris
  int npoints,nlines,ntris;   // count of each
  int nedges;                 // count of unique tri edges
  int maxpoints;              // allocated length of points
  int nsurf;                  // count of lines or tris for 2d/3d

  // ragged 2d arrays for 2d connectivity

  int **neigh_p1;             // indices of other lines connected to endpt 1
  int **pwhich_p1;            // which point (0/1) on other line is endpt 1
  int **nside_p1;             // consistency of other line normal
                              //   SAME_SIDE or OPPOSITE_SIDE
  int **aflag_p1;             // is this line + other line a FLAT,CONCAVE,CONVEX surf
                              //   surf = on normal side of this line
  int **neigh_p2;             // ditto for connections to endpt 2
  int **pwhich_p2;            // ditto for endpt 2
  int **nside_p2;             // ditto for endpt 2
  int **aflag_p2;             // ditto for endpt 2

  // ragged 2d arrays for 3d edge connectivity

  int **neigh_e1;             // indices of other tris connected to edge 1
  int **ewhich_e1;            // which edge (0/1/2) on other tri is edge 1
  int **nside_e1;             // consistency of other line normal
                              //   SAME_SIDE or OPPOSITE_SIDE
  int **aflag_e1;             // is this tri + other tri a FLAT,CONCAVE,CONVEX surf
                              //   surf = on normal side of this tri
  int **neigh_e2;             // ditto for connections to edge 2
  int **ewhich_e2;            // ditto for edge 2
  int **nside_e2;             // ditto for edge 2
  int **aflag_e2;             // ditto for edge 2
  int **neigh_e3;             // ditto for connections to edge 3
  int **ewhich_e3;            // ditto for edge 3
  int **nside_e3;             // ditto for edge 3
  int **aflag_e3;             // ditto for edge 3

  // ragged 2d arrays for 3d corner connectivity

  int **neigh_c1;             // indices of other tris connected to cpt 1
  int **cwhich_c1;            // which corner point (0/1/2) on other tri is cpt 1
  int **neigh_c2;             // indices of other tris connected to cpt 21
  int **cwhich_c2;            // which corner point (0/1/2) on other tri is cpt 2
  int **neigh_c3;             // indices of tris connected to cpt 3
  int **cwhich_c3;            // which corner point (0/1/2) on other tri is cpt 3

  // per-surface 2d/3d connectivity

  struct Connect2d {      // line connectivity

                          // counts, not including self
    int np1,np2;          // # of lines connected to endpts 1/2

                          // pairs of endpoint connections
    int *neigh_p1;        // indices of lines connected to endpt 1
    int *neigh_p2;        // ditto for connections to endpt 2
    int *pwhich_p1;       // which point (0,1) on other line is endpt 1
    int *pwhich_p2;       // ditto for endpt 2
    int *nside_p1;        // consistency of other line normal
    int *nside_p2;        // ditto for endpt 2
                          //   SAME_SIDE = 2 normals are on same side of surf
                          //   OPPOSITE_SIDE = opposite sides of surf
    int *aflag_p1;        // is this line + other line a FLAT,CONCAVE,CONVEX surf
    int *aflag_p2;        // ditto for endpt 2
                          //   surf = on normal side of this line
                          //   aflag = FLAT, CONCAVE, CONVEX
  };

  struct Connect3d {      // tri connectivity

                          // counts, not including self
                          // also not including edge-connected tris for corner pts
    int ne1,ne2,ne3;      // # of tris connected to edges 1,2,3
    int nc1,nc2,nc3;      // # of tris connected to corner pts 1,2,3

                          // pairs of edge connections
    int *neigh_e1;        // indices of tris connected to edge 1
    int *neigh_e2;        // ditto for connections to edge 2
    int *neigh_e3;        // ditto for connections to edge 3
    int *ewhich_e1;       // which edge (0,1,2) on other tri shares edge 1
    int *ewhich_e2;       // ditto for edge 2
    int *ewhich_e3;       // ditto for edge 3
    int *nside_e1;        // consistency of other tri normal
    int *nside_e2;        // ditto for edge 2
    int *nside_e3;        // ditto for edge 3
                          //   SAME_SIDE = 2 normals are on same side of surf
                          //   OPPOSITE_SIDE = opposite sides of surf
    int *aflag_e1;        // is this tri + other tri a FLAT,CONCAVE,CONVEX surf
    int *aflag_e2;        // ditto for edge 2
    int *aflag_e3;        // ditto for edge 3
                          //   surf = on normal side of this tri
                          //   aflag = FLAT, CONCAVE, CONVEX

                          // pairs of corner pt connections
    int *neigh_c1;        // indices of tris connected to corner pt 1
    int *neigh_c2;        // ditto for connections to corner pt 2
    int *neigh_c3;        // ditto for connections to corner pt 3
    int *cwhich_c1;       // which corner pt (0,1,2) on other tri shares corner pt 1
    int *cwhich_c2;       // ditto for corner pt 2
    int *cwhich_c3;       // ditto for corner pt 3
  };

  Connect2d *connect2d;             // 2d connection info
  Connect3d *connect3d;             // 3d connection info

  // struct for storing contact data

  struct ContactSurf {
    int index, neigh_index, type, jflag, nside, ignore_pt;
    double r[3], overlap;
  };

  ContactSurf *contact_surfs;
  int nmax_contact_surfs;

  // data for DumpImage

  int *imflag;
  double **imdata;
  int imax;

  // private methods

  void extract_from_molecule(char *, std::map<std::tuple<double,double,double>,int> *);
  void extract_from_stlfile(char *, int, std::map<std::tuple<double,double,double>,int> *);
  void check2d();
  void check3d();
  void connectivity2d();
  void connectivity3d();
  void check_molecules();
  void stats2d();
  void stats3d();

  void surface_attributes();

  void walk_flat_connections2d(int, int, std::vector<int> *, std::unordered_set<int> *, std::unordered_set<int> *, std::map<int, int> *);

  int modify_param_move(Motion *, int, char **);

  void move_linear(int, int);
  void move_wiggle(int, int);
  void move_rotate(int, int);
  void move_transrotate(int, int);
  void move_rotate_point(int, double *, double *, double, double);
  void move_variable(int, int);
};

}    // namespace LAMMPS_NS

#endif
#endif
