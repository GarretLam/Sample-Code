// Sample C++ code
//
// This code is extracted from one of the solvers based on the CE/SE method in my previous team.
// This solver adopts time marching in the solutions, which are evaluated by the flux conservation in an element, i.e. the basic unit in the method.
// In this sample code, the part with compressible Navier-Stokes Equations solver is demonstrated.
// As demonstration, the codes for other functionalities, information handling and calculation processes are all omitted for clarity.
// The program was designed and mainly developed by myself including the structure.
// The major consideration of this program apart from the solving the targeted equations is speed of calculation.
// Later my supervisor chose it as the major research tool and it was further developed in my previous team.
//
// The program can be run in HPC cluster via hybrid parallelisation.
// 1. Between the computing node: MPI (started from MPI_init)
// 2. Inside the computing node: OpenMP (through directive #pragma omp)
// I chose such approach due to the simplicity of OpenMP and overall effectiveness of such combination.
// It can even achieve about 90% of ideal speed-up on the 150-core cluster in my previous team.
//
// I adopted object-oriented programming (OOP) using the following major features.
// 1. Object features e.g. inheritance - treated an element in the CE/SE method as an object (class Element), which is the basis of the calculation.
//                                     - class Element is built from other smaller classes, which are in turns built on even smaller classes to ensure testability.
//                                     - Demonstrated definitions from Line 180 - 277
// 2. Template - class & functions simplifying the code for different solvers, e.g. class Element.
// 3. Function pointers - simplifying the code for different solving processes, e.g. updateU in Line 153.
// 4. Data structures - grouping calculation parameters, temp variables etc.

int main (int argc, char* argv[]) // Main program of in-house numerical model
{
  SimulationInformation simulationSettings; ExcitationInformation ExcitationParameters;
  int choice = 0, fchoice = 0, schoice = 0, nThreads = 1, mchoice = 0, excitation_choice = 0, excite_EID = -1, option; bool solution_coupling;
  option = getopt(argc, argv, "mn");
  if (option == 'm') { // MPI version is chosen
    User_Menu_mpi(choice, schoice, fchoice, ExcitationParameters, mchoice, solution_coupling, simulationSettings.IsConnectedDomain, simulationSettings.IsOptimizedLoad, simulationSettings.IsSelfDecomposed); // Reading input setting
    simulationSettings.Pm = input_Parameters(fchoice, schoice, ExcitationParameters.excitation_choice);    // Input parameters for simulation
    switch (schoice) { // Choosing the type of simulation, 2D (Omitted here) or 3D, using single computer or a computing cluster with multiple computing nodes
      case 3:
        cese3d_Domain_Decomp(simulationSettings, choice, fchoice, solution_coupling, ExcitationParameters, nThreads); // Three dimensional calculations using parallelization with MPI & OpenMP
        break;
      default:
        cerr << "Error";
        cerr << "\nIncorrect input of simulation module!" << endl;
        exit(EXIT_SUCCESS);
        break;
    }
  }
  else { // This is a single computing node version. Omitted for clarity!
  }
  return 0;
}

void cese3d_Domain_Decomp(SimulationInformation &simulationSettings, int choice, int fchoice, bool &solution_coupling, ExcitationInformation &ExcitationParameters, int nThreads)
{
  switch (simulationSettings.Pm.typeGeometry) { // Choosing the shape of the element
    case 'T': // Tetrahedron shaped elements. Omitted for clarity
      break;
    case 'H': // Hexahedron shaped elements: Demonstration
      switch(simulationSettings.Pm.typePhysics) { // Choosing the type of physical model
        case 'N': // Fluid dynamic solver (solving compressible Navier-Stikes equations)
        {  // Declare all the required variable for Hexahedron shaped elements
          vector<Element<CE_Hex, SE_Hex> > elm; MatrixP Sol_Vec(5,0.0); DummyVar3D<MatrixP, 5, 6> a1; DummyDxy3D<MatrixP, 8, 6> a2; DummyU3DM<MatrixP> a3;
          Hexahedron hexa; E_Info<6,8> info_temp;
          // Below funciton is the calculation
          Cal_Cese3d_Domain_Decomp(elm, simulationSettings, choice, fchoice, solution_coupling, ExcitationParameters, nThreads, Sol_Vec, a1, a2, a3, 6, 8, hexa, info_temp);
        }
          break;
      }
      break;
  }
  return;
}

template <class T1, class T2, class Matrix, class Shape, class DummyContainer, class DummyContainer2, class DummyContainer3, class EInfo>
void Cal_Cese3d_Domain_Decomp(vector<Element<T1, T2> > &elm, SimulationInformation &simulationSettings, int choice, int fchoice, bool &solution_coupling, ExcitationInformation &ExcitationParameters, int nThreads, Matrix Mat, DummyContainer a1, DummyContainer2 a2, DummyContainer3 a3, int NE, int Nnode, Shape element_shape, EInfo info_temp)
{
  // 1. Variables declaration - Demonstration of function pointer adopted in this code (Definitions starts from Line 278)
  InformationForUpdatingSolutionVector<T1, T2, DummyContainer, Matrix> Arguments_U;
  assign_updateU_function(Arguments_U.updateU);       // Assign the updating solution based on the shape of elements
  // Other declarations omitted here
    
  // 2. Initialization of variables and assignment of variables including some function pointers, e.g.
  assign_solver3d(Arguments_U.Solver);            // Assign type of equation solver
  numBCond = new int[simulationSettings.Pm.numBC]; // Set the size of boundary condition array
  read_BC_parameters(Arguments_BC.bcp, bcpartname, bctype_string);    // Read the parameters used for the boundary conditions
  // Code omitted: MPI initialization (Part of code omitted)
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &simulationSettings.rank);    /* get current process id */
  MPI_Comm_size(MPI_COMM_WORLD, &simulationSettings.numberComputeNodes);
    
  // Importing mesh & prepare graph information
  if (simulationSettings.rank == Master) { // Process the global mesh information
    processGlobalMesh3d(element_shape, info_temp, numBCond, simulationSettings.Pm, fchoice, ExcitationParameters, Arguments_BC.bcp, bcpartname, bctype_string, mpiProcess, simulationSettings.numberComputeNodes, index, edges, sizeofEdges, simulationSettings);
    // Code omitted: Send data to other nodes using MPICH3
  }
  else {
    // Code omitted: Receive data from master using MPICH3
  }
  // Set the number of threads to be used in OpenMP for computing nodes individually
  nThreads = omp_get_num_procs();
  omp_set_num_threads(nThreads);
  // Code omitted: Create Graph
  // Code omitted: Build transfer data type and the related set up
  // Each computing node loads its local mesh & create CE/SE elements
  processLocalMesh3d(elm, simulationSettings, CheckPoint, ExcitationParameters.ExcitationElements, ExcitationParameters.PtExcitationElements, InterfaceElement, ExcitationParameters.excitation_choice, rankofDisplayedCheckPt, info_temp, element_shape);
  // Load the initial conditions
  mpi_input_InitialCondition3d(simulationSettings.rank, elm, simulationSettings.Pm, Mat, simulationSettings.Time, simulationSettings.Step, Arguments_BC.sutherlandConst, Arguments_BC.PrandtlNumber, nextOutputTime, simulationSettings.gammaMinus1);    // Input initial conditions
  
  // Sync data across the computing nodes
  prepareDataforTransfer(sendData, elm, InterfaceElement, simulationSettings.rank); // Pack data from SE
  MPI_Neighbor_alltoallv(&sendData[0], sendCounts, sendDisplacements, Solution_type3d, &receiveData[0], receiveCounts, receiveDisplacements, Solution_type3d, comm_graph); // Transfer data
  receiveDatafromTransfer(receiveData, elm, simulationSettings.numberNormalElements, simulationSettings.numberDomainElements, Utemp, simulationSettings.gammaMinus1);  // Copy data to SE
  // Code omitted: Calculate the fluxes of the interface elements
  // Code omitted: Assign initial boundary condition and set up solution copies for later use
    
  // Main calculations loop
  do {
    // Code omitted: Screen output for checking progress & writing results
    // Update the solution vectors and their spatial gradient of all elements 'elm' except the interface elements.
#pragma omp parallel default(none) \
    shared(elm, oldSolution, simulationSettings, ExcitationParameters) \
    firstprivate(Arguments_U, Arguments_SpatialGradient)
    {
      cese3d_core(simulationSettings, Arguments_U, Arguments_SpatialGradient, ExcitationParameters, elm, oldSolution);
    }
    // Code omitted: Sync data across the computing nodes
    // Update spatial gradient of interface cell
#pragma omp parallel default(none) \
    shared(elm, InterfaceElement, oldSolution, simulationSettings) \
    firstprivate(Arguments_SpatialGradient, ne_idx)
    {
      updateSpatialGradient(simulationSettings, InterfaceElement, ne_idx, Arguments_SpatialGradient, elm, oldSolution);
    }
    // Code omitted: Sync data across the computing nodes after updating spatial gradient of all interface elements
    // Code omitted: Update the fluxes of all interface elements
    // Post-processing - Apply Boundary Conditions & copy updated results to old results for the calculation in the next time step
#pragma omp parallel default(none) \
    shared(simulationSettings, elm, oldSolution) \
    firstprivate(Arguments_BC)
    {
      cese3d_postprocessing(simulationSettings, elm, Arguments_BC, oldSolution);
    }
    simulationSettings.Step++; // Updating step count
    simulationSettings.Time += simulationSettings.Pm.timeInc; // Updating calculation time
  } while (simulationSettings.Time < simulationSettings.Pm.totalTime);

  // Code omitted: Clear up all used pointers
  MPI_Finalize();
  return;
}

// cese3d core function: major part of solution updating
template <class CE, class SE, class Matrix, class DummyContainer, class DummyContainer2, class DummyContainer3>
void cese3d_core(SimulationInformation &simulationSettings, InformationForUpdatingSolutionVector<CE, SE, DummyContainer, Matrix> &Arguments_U, InformationForUpdatingSpatialGradient<CE, SE, DummyContainer2, DummyContainer3, Matrix> &Arguments_SpatialGradient, ExcitationInformation &ExcitationParameters, vector<Element<CE, SE> > &elm, SolutionSet<Matrix> &oldSolution)
{
  // Update solution vector using function pointer, which calls updateU_Hexahedron in this demonstration)
  Arguments_U.updateU[simulationSettings.shape](elm, simulationSettings.Pm, Arguments_U.dummy, simulationSettings.numberNormalElements, simulationSettings.Pm.ucTemperatureCoef, simulationSettings.gammaMinus1, simulationSettings.gamma, Arguments_U.Solver);
#pragma omp single
  {
    apply_excitation(Arguments_U.Utemp, simulationSettings.Pm, elm, ExcitationParameters, simulationSettings.Time, simulationSettings); // apply excitation to the elements if needed
  }
  // Update spatial gradient of solution vector using function pointer
  Arguments_SpatialGradient.updateUxUyUz[simulationSettings.shape](simulationSettings.numberNormalElements, elm, Arguments_SpatialGradient.dumU, Arguments_SpatialGradient.dummyCE, simulationSettings.Pm.timeInc, Arguments_SpatialGradient.co, simulationSettings.gammaMinus1, simulationSettings.gamma, oldSolution.Uo, oldSolution.Uxo, oldSolution.Uyo, oldSolution.Uzo, Arguments_SpatialGradient.alpha, Arguments_SpatialGradient.cal_locCFL);
}

template <class CE, class SE, class DummyContainer> // Function for calculating the fluxes and the solution at (n+1)-th time level of the hexahedron element
void updateU_Hexahedron(vector<Element<CE, SE> > &elm, const Parameter &Pa, DummyContainer &a, int N_elem, const double &ucTempCoef, const double &gammaMinus1, const double &gamma, void (*Sptr2[numberSolverType3d])(Element<CE, SE> &elm, const Element<CE, SE> &ne1, const Element<CE, SE> &ne2, const Element<CE, SE> &ne3, const Element<CE, SE> &ne4, const Element<CE, SE> &ne5, const Element<CE, SE> &ne6, const Parameter &Pm, DummyContainer &a))
{
#pragma omp for
  for (long int k = 0; k < N_elem; ++k) {
    // 1. Update Ut at old time step
    elm[k].calFGH(a.F, a.G, a.H);
    elm[k].updateFluxDerivative(ucTempCoef, gammaMinus1, gamma, a);
    // 2. Calculate the flux of each basic conservation element using function pointer
    Sptr2[Pa.typeSolver](elm[k], elm[elm[k].EInfo().getAdjElem(0)-1], elm[elm[k].EInfo().getAdjElem(1)-1], elm[elm[k].EInfo().getAdjElem(2)-1], elm[elm[k].EInfo().getAdjElem(3)-1], elm[elm[k].EInfo().getAdjElem(4)-1], elm[elm[k].EInfo().getAdjElem(5)-1], Pa, a);
  }

#pragma omp for
  for (long int k = 0; k < N_elem; ++k) {
    elm[k].updateU(elm[elm[k].EInfo().getAdjElem(0)-1].getFluxBCE(elm[k].EInfo().getAdjElemPos(0)), elm[elm[k].EInfo().getAdjElem(1)-1].getFluxBCE(elm[k].EInfo().getAdjElemPos(1)),elm[elm[k].EInfo().getAdjElem(2)-1].getFluxBCE(elm[k].EInfo().getAdjElemPos(2)), elm[elm[k].EInfo().getAdjElem(3)-1].getFluxBCE(elm[k].EInfo().getAdjElemPos(3)), elm[elm[k].EInfo().getAdjElem(4)-1].getFluxBCE(elm[k].EInfo().getAdjElemPos(4)), elm[elm[k].EInfo().getAdjElem(5)-1].getFluxBCE(elm[k].EInfo().getAdjElemPos(5)), elm[k].getVolume(), gammaMinus1);
  }
}

template < class CE, class SE >
class Element: public CE, public SE // Class Element definition
{
  public:
    Element();
};
// Class Element instantiation
template < class CE, class SE >
Element< CE, SE >::Element() {}
template class Element<CE_Tet, SE_Tet>;
template class Element<CE_Hex, SE_Hex>;
// Other CEs & SEs are omitted

class SE_Hex // Solution element for hexahedron element, sample code
{
  public:
    SE_Hex();
    void calFGH(double F[], double G[], double H[]);                // Calculate Inviscid flux F, G & H
    void updateFluxDerivative(const double &ucTempCoef, const double &gammaMinus1, const double &gamma, DummyVar3D<MatrixP, 5, 6> &a); // Update the flux derivative of flow variables
    void calFluxBCE(const CE_Hex &elm, const CE_Hex &ne1, const CE_Hex &ne2, const CE_Hex &ne3, const CE_Hex &ne4, const CE_Hex &ne5, const CE_Hex &ne6, const double &timestep, DummyVar3D<MatrixP, 5, 6> &a);
    void updateU(const MatrixP &Flux1, const MatrixP &Flux2, const MatrixP &Flux3, const MatrixP &Flux4, const MatrixP &Flux5, const MatrixP &Flux6, const double &Vol, const double &gammaMinus1);
    const MatrixP &getU() const;                                    // Get the solution vector U
    // Omitted some functions.
  private:
    MatrixP U, Ux, Uy, Uz, Ut;
    // Omitted some variables for clarity
 };

class CE_Hex: public ceGeometry_Hex
{
  friend class SE_Hex;

  public:
    CE_Hex();
    void setDomainType(int type);
    const int &getDomainType() const;
    const double &CFL() { return locCFL; };
    // Omitted some functions about calculating spatial gradients of the element
    
  private:
    double locCFL;
    int domain_type;
};

class ceGeometry_Hex
{
  public:
    ceGeometry_Hex();
    void setGeometry(Vec3d, Vec3d, Vec3d, Vec3d, Vec3d, Vec3d, Vec3d, Vec3d,
                     Vec3d, Vec3d, Vec3d, Vec3d, Vec3d, Vec3d);
    const Vec3d &getSolPt() const;
    void setEInfo(E_Info<6,8> &eInfo);     // Setup information of element
    const E_Info<6,8> &EInfo() const;
    // Omitted some functions about returning spatial information for flux calculation.
    
  private:
    Vec3d hexCentroid, SolutionPt;
    DiPyramid BCE[6];
    E_Info<6,8> eHex;
    // Omitted some variables
};

class TriDipyramid
{
  public:
    TriDipyramid();
    TriDipyramid(Vec3d&, Vec3d&, Vec3d&, Vec3d&, Vec3d&);
    void setTDPyramid(Vec3d, Vec3d, Vec3d, Vec3d, Vec3d);
    const double &getVolume() const;            // Get the volume
    const Vec3d &getCentroid() const;            // Get the centroid
    
  private:
    Vec3d Centroid;
    double tdpVol;
};

template < int m, int n>
class E_Info
{
  friend class CE_Tet;
  
  public:
    E_Info();                       // asize = no. of adj. elements, nsize = no. of nodes
    void setElementID(int);                                        // Set the ID number of the element
    void setAdjElement(int, int);                                // Set the ID number of the adjacent elment at mth position
    const int &getElementID() const;                                // Get the ID number of the element
    const int &getAdjElem(int) const;                                // Get the ID number of the adjacent elment at mth position
    // Omitted some functions about returning element information.
    
  private:
    int asize, nsize;
    int Element_ID;
    int AE_ID[m];                // Adjacent Element ID
    int Node[n];                // Node number of vertex
    // Omitted some variables for clarity
};
// Class instantiation. Other instantiations are omitted.
template class E_Info<6,8>;

// Function pointer
// Definition of function pointer for updating solution based on the shape of element
const int numberCEType3d = 2;
template <class CE, class SE, class DummyContainer>
struct updateU_type3d
{
  typedef void (*updateU_ptr[numberCEType3d])(vector<Element<CE, SE> > &elm, const Parameter &Pm, DummyContainer &a, int N_elem, const double &ucTempCoef, const double &gammaMinus1, const double &gamma, void (*Sptr2[numberSolverType3d])(Element<CE, SE> &elm, const Element<CE, SE> &ne1, const Element<CE, SE> &ne2, const Element<CE, SE> &ne3, const Element<CE, SE> &ne4, const Element<CE, SE> &ne5, const Element<CE, SE> &ne6, const Parameter &Pm, DummyContainer &a));
};

// Data structures containing the function pointer
template <class CE, class SE, class DummyContainer, class Matrix>
struct InformationForUpdatingSolutionVector
{
  typename updateU_type3d<CE, SE, DummyContainer>::updateU_ptr updateU;
  // Other variable omitted here.
};

// Assign the function pointer for updating solution based on the shape of element
template <class CE, class SE, class DummyContainer>
void assign_updateU_function(void (*updateU_ptr[numberCEType3d])(vector<Element<CE, SE> > &elm, const Parameter &Pm, DummyContainer &a, int N_elem, const double &ucTempCoef, const double &gammaMinus1, const double &gamma, void (*Sptr2[numberSolverType3d])(Element<CE, SE> &elm, const Element<CE, SE> &ne1, const Element<CE, SE> &ne2, const Element<CE, SE> &ne3, const Element<CE, SE> &ne4, const Element<CE, SE> &ne5, const Element<CE, SE> &ne6, const Parameter &Pm, DummyContainer &a)))
{
  updateU_ptr[0] = updateU_Tetrahedron<CE, SE, DummyContainer>;
  updateU_ptr[1] = updateU_Hexahedron<CE, SE, DummyContainer>;
}
