\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amsfonts}
\usepackage{url}
\usepackage{graphicx}
\usepackage[superscript]{cite}

\title{
\vspace{-3em}
Proposal: Neurons as bandit arms in ablation studies}
\vspace{-1em}
\author{Reuben Narad}
\date{}

\begin{document}
\vspace{-2em}

\maketitle\vspace{-2em}

\vspace{-1em}

\subsection*{Context: Neural TSP Solver + Mechanistic interpretability}
I'm currently doing interpretability work on a deep RL TSP solver. The model of interest is a transformer pointer network\cite{kool2019} trained with policy gradient, to construct TSP solutions. One forward pass takes node embeddings and partial tour, and outputs next-node probabilities. 

To interpret this model, we follow OpenAI\cite{gao2024scaling} and train a Sparse Auto Encoder (SAE), an encoder/decoder model that learns an overcomplete but sparse representation of the TSP solver's activations. This SAE's neurons, called features, represent a direction in the model's representation space. Due to sparsity, they appear monosemantic\cite{polysemantic} and interpretable in our setting. Figure \ref{fig:four_images} shows visualizations of interesting features we found. For more detail, see the project demo \cite{narad2024}. To see similar work being done on LLMs and  get a feel for what "features" mean, check out Neuronpedia\cite{lin2024neuronpedia}!  

\begin{figure}[h]
    \centering
    \begin{tabular}{ccccc}
        \includegraphics[width=.2\textwidth]{1.png} &
        \includegraphics[width=.2\textwidth]{2.png} &
        \includegraphics[width=.2\textwidth]{3.png} &
        \includegraphics[width=.2\textwidth]{4.png} &
        \includegraphics[height=.2\textwidth]{5.png} \\
        "I'm on the edge" & "Focus on one spot" & "Linear separator" & Unclear...? &
    \end{tabular}
    \caption{Representative examples of feature themes that we found in the TSP solver model}
    \label{fig:four_images}
\end{figure}
\vspace{-2em}

\subsection*{Proposal: Treat SAE features like bandit arms}

To test hypotheses about the causal relationship between these features and the model's final behavior, researchers often run \textbf{ablation studies}, in which the feature of interest's direction is zeroed out in the model's activation. Then, you observe the model's behavior compared to a baseline. For thousands of features, this can be expensive, and is a pain point for interp reserachers.

My idea is that a bandit-style best arm identification algorithm could be used to speed up the feature ablation search process. 
Given the set of features, we are interested in finding the top-p features-- arms who collectively contribute at least $p$ portion of the total metric of interest (akin to nucleus sampling).
Solving an instance with a feature ablated is like pulling the arm, observing reward as impact on attribute of interest.
Thus, feaures that show no impact are filtered out.

The first hypothesis I'd like to test is \emph{mimicking a nearest neighbors policy}, something I've suspected the model is often doing. At every step of the tour, identify the nearest neighbor, and extract the probability assigned to it by the both ablated and unablated model. Take the average of the difference of the two over the trajectory as the reward for the arm pull. 

\vspace{-1em}
{\footnotesize
\bibliographystyle{unsrt}
\bibliography{references}
}
\end{document}