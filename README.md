# Udacity_Banana_Unity
Udacity Project Navigation Report

Introduction
In this document I will cover the explanation and description of my solution to The Challenge project Navigation for the Deep Reinforcement Learning Nanodegree of Udacity. My solution covers 7 different algorithms as I wanted to explore all possible improvements to the Vanilla deep RL DQN algorithm. The skeleton of this solution is based on the coding Exercise Deep Q-Networks (lesson 2) of this program, while I also use other resources as books, or public information available that I will detail on the references.
The application solves the environment with the following 7 implementations
Mode 1  Plain Deep DQN vanilla. (Greedy algo for action selection)
Mode 2  Duelling DQN with priority buffer replay (Greedy algo for action selection)
Mode 3  Duelling DQN without priority buffer replay (only replay buffer) (Greedy algo for action selection)
Mode 4  categorical DQN, without priority buffer replay (only replay buffer) (Greedy algo for action selection)
Mode 5 Duelling DQN, with priority buffer replay and Noisy Layer for exploration. (NO greedy algo for action selection)
Mode 6  DQN n-steps (only replay buffer) (Greedy algo for action selection)
Mode 7  Rainbow DQN (Duelling DQN + n-Steps + Categorical + Noisy Layer for Exploration + priority Buffer Replay)
Installation
My solution works as an application which run in a windows command line window (I did not try in Linux, but I suspect that with minimum changes it will work). To setup the environment, I simply setup the DRLND GitHub repository in an Conda environment as is demanded in the project instructions and then a windows(64-bit) unity environment. I use Pycharm Professional for code Development:
Just copy and paste from Udacity
Step 1: Clone the DRLND Repository
________________________________________
If you haven't already, please follow the instructions in the DRLND GitHub repository to set up your Python environment. These instructions can be found in README.md at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.
(For Windows users) The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

Step 2: Download the Unity Environment
________________________________________
For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:
•	Linux: click here
•	Mac OSX: click here
•	Windows (32-bit): click here
•	Windows (64-bit): click here
Then, place the file in the p1_navigation/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.
(For Windows users) Check out this link if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.
(For AWS) If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use this link to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to enable a virtual screen, and then download the environment for the Linux operating system above.)
