+++
title = 'Automatic crossword generation using LLM Agents'
date = 2024-06-15
math = true
+++

In this post, I will detail how I created a NLP-based automatic crossword puzzle generator, which takes in any topic as user input, and generates answer-clue pairs relating to that topic along with a crossword board automatically. This is based on the [AgentCoder](https://arxiv.org/abs/2312.13010) approach.

## Demo
My source code can be found [here](https://github.com/jlohding/xword).

1. User inputs topic: 'sports'
```python
python main.py sports
```
2. Crossword is generated, with board, answers, and clues:
```text
User Input Topic: SPORTS
8 out of 14 words generated used

F - G A M E - - - - 
O - - T - - - - - - 
O - - H O C K E Y - 
T - - L - O - - - - 
B A S E B A L L - - 
A - - T - C - - - - 
L - - I - H O S T - 
L - - C - - - - V - 
- - - - - - - - - - 
- - - - - - - - - - 

ACROSS:
(1, 3) - A contest of risks (4)
(3, 4) - Sport with sticks and pucks (6)
(5, 1) - America's pastime, batting around (8)
(7, 6) - One who greets at the door (4)
DOWN:
(1, 1) - Sport involving goals with kicks (8)
(1, 4) - Fit for sports, sounds like a competition (8)
(3, 6) - Mentor of teams (5)
(7, 9) - Small screen box (2)
```

## Motivation
While working at Julius Baer, we worked on replicating the results and implementing the framework in [AgentCoder](https://arxiv.org/abs/2312.13010), which is the current state-of-the-art on HumanEval and MBPP. 

The idea behind AgentCoder is simple: By using a multi-agent system framework, where multiple autonomous agents interact with each other, general tasks can be broken down into more granular steps, which yields significant benefits over single-agent approaches. In their paper, they break down the task of solving programming problems into the subtasks of generating test cases, generating code, debugging code, and executing code. 

## Methodology
### Generating answers
Before we use the agentic approach to generate clues, we need to generate crossword answers from any given input topic. This is not a complex task, and we don't need an LLM for this. Instead, we use static embeddings (in this case GloVe), and find the top-k words with the most similar embeddings to a given input topic. The idea is that words related to a given topic are likely to have similar representations in embedding vector space.

For example, given the word `apple`, pre-trained GloVe embeddings give us the following words as having the most similar embeddings:
```text
iphone, ipad, chip, pc, intel, ibm, android, product, dell, cola, desktop, amd
```

We will then use these as answer words, and generate clues for them.

### Generating clues
Applying the agent approach to crossword clue generation, we break down the task into generating clues, guessing answers from clues, and improving clues. Given an `answer`, we attempt to generate a cryptic crossword clue using the following pipeline:

![xword Pipeline](/img/agent_pipeline.png#center)

Example:
1. Choose an `answer: iPad`; our goal is to generate a clue for this answer
2. Clue Agent: Given `answer: iPad`, it generates the `clue: Apple's touch device`
3. Guess Agent: Given the clue, it guesses the answer `iPod`.
    - Since this is wrong, we pass `clue: Apple's touch device`, and `answer: iPad` to the Debug Agent
    - The debug agent improves on the clue, and generates a new `clue: A tablet that's not made of stone`
4. Guess Agent: Given the new clue, it correcly guesses the answer `iPad`.
5. Final output: `{answer: iPad, clue: A tablet that's not made of stone}`.

## Single agent performance
It is interesting to find out what the performance of simply using a single agent to generate clues is: This serves to validate the need for a multi-agent system. 

Since it is difficult to come up with an evaluation metric for this task, we rely on empirical observations. Given an input topic `SPORTS`, we compare the one-shot and the agentic framework results. More work needs to be done to validate this approach but here are some examples:

```text
One-shot: ('SOCCER', "Golfer's cousin kicks and scores (6)")
Agent: ('SOCCER', 'Game with goals and nets (6)')

One-shot: ('BASKETBALL', 'Net asset? (10)')
Agent: ('BASKETBALL', 'Hoop pursuit? (10)')
```
 
It is also interesting to note that the divergence in clue quality becomes more apparent when using less powerful models, since the one-shot performance is likely worse.

## Future work
More work needs to be done to improve on the Debugging Agent, and evaluation metrics need to be set clearly for us to decisively prove that the agentic framework is useful. Also, I intend to make the project an actual playable crossword game with a GUI sometime in the future.