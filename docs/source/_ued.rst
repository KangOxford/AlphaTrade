.. _ued:

Emergent Complexity and Zero-shot Transfer via
====================

1. Authors: 
--------------------

Michael Dennis, Natasha Jaques, Eugene Vinitsky, Alexandre Bayen, Stuart Russell, Andrew Critch, Sergey Levine

2. Affiliation: 
--------------------

University of California Berkeley AI Research (BAIR), Berkeley, CA, 94704 

3. Keywords: 
--------------------

Reinforcement Learning, Unsupervised Environment Design, Emergent Complexity, Zero-shot Transfer, Protagonist Antagonist Induced Regret Environment Design (PAIRED)

4. Urls: 
--------------------

arXiv:2012.02096v2 [cs.LG] 4 Feb 2021, Github: None

5. Summary: 
--------------------

(1): This article presents a new methodology for designing environments in reinforcement learning, particularly addressing problems like transfer learning, unsupervised learning and emergent complexity, which all require creating a distribution of tasks or environments to train a policy. 

(2): Previous approaches to automatically generate environments have limitations: domain randomization is unable to generate structure or adapt the difficulty of the environment, while minimax adversarial training often led to unsolvable worst-case environments. To solve this, the authors propose Protagonist Antagonist Induced Regret Environment Design (PAIRED) which includes a second, allied agent that is used to generate structurally complex yet solvable environments. 

(3): In this article, the authors present their novel PAIRED methodology for designing environments through unsupervised environment design. They introduce an adversary that attempts to maximize the regret between the protagonist and antagonist agents, which generates environments that the PAIRED agent can navigate through. 

(4): The authors demonstrate that PAIRED generates a natural curriculum of increasingly complex environments that enables better zero-shot transfer performance when tested in novel environments. The approach shows promising results on multiple RL tasks and supports the goal of unsupervised environment design.

6. Methods: 
--------------------

(1): The article presents a new methodology for designing environments in reinforcement learning called Protagonist Antagonist Induced Regret Environment Design (PAIRED), which addresses problems such as transfer learning, unsupervised learning, and emergent complexity. 

(2): PAIRED includes a second, allied agent that generates structurally complex yet solvable environments by attempting to maximize the regret between the protagonist and antagonist agents. Previous approaches like domain randomization and minimax adversarial training have limitations, such as the inability to generate structure or adapt difficulty and unsolvable worst-case environments, respectively. 

(3): The authors demonstrate that PAIRED generates a natural curriculum of increasingly complex environments that allows for improved zero-shot transfer performance on novel environments. The methodology was tested on multiple RL tasks, including a modified version of the MuJoCo hopper domain, and supports the goal of unsupervised environment design. 


7. Conclusion:
--------------------

(1): The significance of this work lies in its introduction of a novel methodology, Protagonist Antagonist Induced Regret Environment Design (PAIRED), for designing environments in reinforcement learning that address transfer learning, unsupervised learning, and emergent complexity. PAIRED generates increasingly complex environments that enable better zero-shot transfer performance in novel environments, and thus supports the goal of unsupervised environment design. 

(2): Innovation point: The authors introduce a new methodology (PAIRED) that overcomes limitations of previous approaches to environment design in reinforcement learning. (3): Performance: The authors demonstrate promising results in multiple RL tasks, including improved zero-shot transfer performance, which supports the potential application of the PAIRED methodology to real-world scenarios. (4): Workload: The workload for implementing the PAIRED method may be higher than for traditional reinforcement learning methods, as it requires training a second, allied agent and designing an environment generation process. However, the potential benefits in reduced manual environment design and improved performance may outweigh this additional workload.

