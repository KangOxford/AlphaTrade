**Meeting Record** 
</br>`Wednesday Meeting` at the `S0.29`, with `Prof. Hambly`, `Dr. Christian`, and `Prof. Foerster` at `Mathematical Institute`. 
</br> <ins>Still confused about the underlined parts</ins>

**In Summary** 
1. We have what exactly was the environment that only uses historical data without modelling any response from other agents to our actions for now.
2. And then we are going to give the agent the ability to eat into the order book, to excute the trades.
3. But we are still going to use the fact that there are changes to the order book based on the incoming orders.
4. So if I eat into the order book, and somebody else happens to like, throw more orders, I can keep excuting. <ins>So the arrival of new orders into the order book is going to be assumed to be unchanged.</ins>
5. GPU simulation already built: turn order flow into order book.
6. <ins>Then I can add to the order flow as the agent</ins> The agent can add to the order flow, and then we should be good because just pretend that the historical flow is the agent. Then gradually, we could replace more agents with <ins>it</ins> to have the response. <ins>And that gives us stays keeps us close to the data.</ins>

**In Details**
1. It isn't an interesting way to generate the environment, basically, it's basically the environment actually consists of real data.
2. Some heuristic for how the real data changes based on the interactions of the agent. So it's basically saying, and we don't need to make it because that was a linearity assumption. 
3. We don't really have to go and generate all the data. We just want to generate a difference between what happened in it.
4. However, if we interact in the market, there's going to be a tiny $\delta$, we are just going to try and like have a model for the delta between the data based upon the agents actions, as a result, and you've done this in your in your paper, right, because you've said there is a linear there is a delta is delta that is linear. 
5. We only have to have a model for the perturbation of the data. And obviously, that means that the overall <ins>approximation error, because I have to first order in correct</ins>. And if I'm like one of many people playing in the market, hopefully the market isn't gonna look completely different based on my actions.
6. We then have to be sure that we have enough real data so that we can <ins>produce enough samples and the agent doesn't just memorise these trajectories.</ins>
7. If you take out a big chunk, then people are likely to throw things in because they think they're gonna get executed.
8. <ins>There's no approximations at all.</ins>
9. For an agent: observation $\tau$ is the 10 minites order book history. And also dollar: how much you want to sell.
10. The reward is how much money we get for this.
11. The action space is in the next 10 minutes, at any time point we can take an action. From execution, we have determined a period to excute.
12. At the end of 10 minutes, everything else will get sold. We actually have to clean the position.
13. Challenge: Do we have enough data?
14. And then later on, maybe once you have this agent, you can throw a bunch of these into the environment that want to buy and sell different amounts of stuff. And that generates response.
15. Then you have a model for response among these RL agents but always <ins>grounded around the historical data<ins>. <ins>And the response emerges quite naturally because there's other agents</ins>.
