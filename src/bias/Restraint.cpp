/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2011-2015 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed-code.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "Bias.h"
#include "ActionRegister.h"


using namespace std;


namespace PLMD{
namespace bias{

//+PLUMEDOC BIAS RESTRAINT
/*
Adds harmonic and/or linear restraints on one or more variables.  

Either or both
of SLOPE and KAPPA must be present to specify the linear and harmonic force constants
respectively.  The resulting potential is given by: 
\f[
  \sum_i \frac{k_i}{2} (x_i-a_i)^2 + m_i*(x_i-a_i)
\f].

The number of components for any vector of force constants must be equal to the number
of arguments to the action.

Additional material and examples can be also found in the tutorial \ref belfast-4 

\par Examples
The following input tells plumed to restrain the distance between atoms 3 and 5
and the distance between atoms 2 and 4, at different equilibrium
values, and to print the energy of the restraint
\verbatim
DISTANCE ATOMS=3,5 LABEL=d1
DISTANCE ATOMS=2,4 LABEL=d2
RESTRAINT ARG=d1,d2 AT=1.0,1.5 KAPPA=150.0,150.0 LABEL=restraint
PRINT ARG=restraint.bias
\endverbatim
(See also \ref DISTANCE and \ref PRINT).

*/
//+ENDPLUMEDOC

class Restraint : public Bias{
  std::vector<double> at;
  std::vector<double> kappa;
  std::vector<double> slope;
  Value* valueBias;
  Value* valueForce2;
public:
  explicit Restraint(const ActionOptions&);
  void calculate();
  static void registerKeywords(Keywords& keys);
};

PLUMED_REGISTER_ACTION(Restraint,"RESTRAINT")

void Restraint::registerKeywords(Keywords& keys){
   Bias::registerKeywords(keys);
   keys.use("ARG");
   keys.add("compulsory","SLOPE","0.0","specifies that the restraint is linear and what the values of the force constants on each of the variables are");
   keys.add("compulsory","KAPPA","0.0","specifies that the restraint is harmonic and what the values of the force constants on each of the variables are");
   keys.add("compulsory","AT","the position of the restraint");
   componentsAreNotOptional(keys);
   keys.addOutputComponent("bias","default","the instantaneous value of the bias potential");
   keys.addOutputComponent("force2","default","the instantaneous value of the squared force due to this bias potential");
}

Restraint::Restraint(const ActionOptions&ao):
PLUMED_BIAS_INIT(ao),
at(getNumberOfArguments()),
kappa(getNumberOfArguments(),0.0),
slope(getNumberOfArguments(),0.0)
{
  parseVector("SLOPE",slope);
  parseVector("KAPPA",kappa);
  parseVector("AT",at);
  checkRead();

  log.printf("  at");
  for(unsigned i=0;i<at.size();i++) log.printf(" %f",at[i]);
  log.printf("\n");
  log.printf("  with harmonic force constant");
  for(unsigned i=0;i<kappa.size();i++) log.printf(" %f",kappa[i]);
  log.printf("\n");
  log.printf("  and linear force constant");
  for(unsigned i=0;i<slope.size();i++) log.printf(" %f",slope[i]);
  log.printf("\n");

  addComponent("bias"); componentIsNotPeriodic("bias");
  addComponent("force2"); componentIsNotPeriodic("force2");
  valueBias=getPntrToComponent("bias");
  valueForce2=getPntrToComponent("force2");
}


void Restraint::calculate(){
  double ene=0.0;
  double totf2=0.0;
  double fact=1.0;
  for(unsigned i=0;i<getNumberOfArguments();++i){
    const double cv=difference(i,at[i],getArgument(i));
    const double k=kappa[i];
    const double m=slope[i];
    const double f=-(k*cv+m);
    //if(getPntrToArgument(i)->isEnsemble()) fact = 1./static_cast<double>(getPntrToArgument(i)->getEnsemble());
    ene+=0.5*fact*k*cv*cv+m*cv;
    setOutputForce(i,f);
    totf2+=f*f;
  };
  valueBias->set(ene);
  valueForce2->set(totf2);
}

}


}
