
## SASAV: Self-Directed Agent for Scientific Analysis and Visualization

 Agentic AI to automate the initial analysis and viuslaizaiton for scientific data.

![Autonomous Agent](assets/autonomy.png)

### 1. Required Packages
#### OpenAI Python API library and OpenAI Agents SDK
```bash
pip install openai openai-agents
```
#### Langchain related packages
```bash
pip install langchain-chroma langchain-openai langchain-community langchain-huggingface
```
#### Others
```bash
pip install dearpygui pymupdf4llm tiktoken ipython
```

### 2. Visualization Tools
Visualization tools for agent to call as needed, "vtk" folder.

####  Dependencies
- C++11 or higher compiler.
- [VTK](https://github.com/Kitware/VTK), The Visualization Toolkit (VTK).
- [MPI](http://www.mpich.org)

#### Build
1. Install VTK
    * Follow the [instruction](https://docs.vtk.org/en/latest/build_instructions/index.html) to build VTK
    * Another use full resource can be found [here](https://www.cs.purdue.edu/homes/xmt/classes/CS530/spring2018/project0.html)
2. TODO: Get and build example code
```bash
git clone https://github.com/sunjianxin/vtk
cd vtk
mkdir build
cd build
cmake ..  \
-DCMAKE_CXX_COMPILER=mpicxx \
-DVTK_DIR:PATH=path_to_VTK_DVR-MFA_installation_folder
```
*path_to_mfa_include_folder* is the folder location in the project folder in step 2. *path_to_VTK_DVR-MFA_installation_folder* is the installation location when you configure VTK_DVR-MFA before building, and it is normally at */usr/local/include/vtk-version* by default.