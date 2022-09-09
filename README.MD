# MathNFL-PowerRank
I have never done something like this. I would like to apply various quantitative approaches to better understand NFL results and create power rankings. My first ideas are
- Applying Elo to measure skill level of teams.
- A Bayesian approach to latent skill variable.

Below I explain each of these approaches, the results will be below in a table.

Eventually I would like to evaluate these results against betting lines and my own thoughts. I will try and record what I think the win probability for each game is every week.

---
## Elo
The information on Elo rating system comes from the respective [Wikipedia page](https://en.wikipedia.org/wiki/Elo_rating_system). Here each team $i$ will have a rating $R_i$. I follow the Elo suggested scaling, namely that a difference of 200 points in 2 teams ratings should yield an expected score of 0.75, this means that the team has a 75% chance of winning (if we ignore draws). In reality we will account for draws with a draw being worth 0.5 wins. Hence the expected score for team $j$ when playing against team $j$ will be
$$E_i = \frac{1}{1 + 10^{(R_j - R_i)/400}}.$$
The rank update will occur after each week and works as follows. A team $i$ shall either achieve 1, 0.5, or 0 points based on realized results each week (by winning, drawing, or losing respectively, denoted by $S_i$. The difference between this score and their expected score will be multiplied by a factor $K$, finally this adjustment will be added to their original score.
$$R_i^{\textrm{new}} = R_i^{\textrm{old}} + K (S_i - E_i).$$
$K$ represents the maximal possible adjustment per game, we will experiment with different values here, chess has used 16 for masters and 32 for noobs.
Each team will start out with a ranking of 1000, and each week their ranking will be updated to tell us where they stand. Furthermore a predicted winner of each match will be calculated.

Note that we do not take much into account here, things such as home field advantage are not looked after at all. At week 1, each team will have a equal probability to win or lose, so it may take a few weeks for the ranks to establish themselves. To address this, I will also run a simulation on the previous (2021/22) season. From here 2 different rankings will stem. 
- Each team starts this season (2022/23) out with last season's rank.
- Each team starts this season (2022/23) out with a contracted version of last seasons rank (a sort of reset to account for teams changing).


## Bayes
While Elo is an easy to implement and understand model, I am an appreciator of Bayesian statistics. So it only makes sense that I try to apply it here. I will preface everything that follows, that I really do not know what I am doing here with applying mathematical models to sports, and am just trying to learn and play with some ideas. Also all my homies hate frequentist statistics, so Bayes all the way baby.

### Intro to Bayes
If you know about Bayesian statistics, this will probably not be helpful to you, or if you just don't care. Also I tried to use very little formulas, and without a chalkboard, that is not easy.

First, a quick lesson on Bayesian approaches ([here](https://en.wikipedia.org/wiki/Bayesian_statistics) is a proper one). Essentially we start out any situation with our prior beliefs (for instance a certain aboreal YouTuber may believe that the Steelers are going to the Superbowl). We quantify these beliefs in terms of a probability distribution. Then upon observing an event, we update our beliefs according to how likely we quantified the event to be.

The following is for demonstrative purposes only, and not actually how the model works. I will take my Giants as an example, say we want to know on average how many of their first 4 games they will win, call this $\mu$. Now I have my beliefs about $\mu$, these are represented by the prior distribution. 

Now the actual amount of games they win is something random around the average, for simplicity say they they have a probability $p$ of winning each game, so the probability that they win all 4 would be given by $p^4$ or the probability that they lost all 4 would be given by $(1-p)^4$.

The Giants play the Titans, Panthers, Cowboys, and Bears. So I would say there is about a 5% chance they win 4 games, a 25% chance they win 3, 50% they win 2, 20% they win 1, and 5% they win 0. Now I have my 'prior' distribution. Note that before we said that each game had a probability $p$ for the Giants to win, and didn't take into account who the opponents were, so we shouldn't have either, but this is an example and shouldn't be taken that seriously. 

Now for a given $\mu$ we can actually find the probability $p$, just take average amount of wins $\mu$ and divide it by the number of games (4), to find the probability $p$ for this given $\mu$. 

After week 4, I will have a realization of how many times the Giants won (hopefully it isn't 0) that was partially due to the value of $\mu$. Remember, we **don't** know the value of $\mu$ and only have some belief. Now for every possible value of $\mu$ we evaluate what the likelihood of our observation was, and call this $f(x | \mu)$. Say the Giants won all 4 games, if $\mu$ was equal to 4, then $p = 4/4 =1$, so $p^4 = 1^4 =1$, and this had a 100% chance of happening (in math terms $f(x=4| \mu=4) = 1$). The story isn't so simple for another value of $\mu$, say 3. We get that $p = \frac{3}{4}=0.75$, then the likelihood to have won 4 games is $\binom{4}{3}0.75^3(1-0.75) \approx 0.42$. The wierd term at the beginning is the binomial coefficient and just deals with how many different ways you can win 3 out of 4 games, and all that is relevant here is that we have that the likelihood $f(x=4 | \mu=3) = 0.42$ of what we observed (4 wins) given the wins on average (3), is about 42%. The model will continue to calculate the likelihoods for $\mu = 0,1,2$, but that would be a pain (this is why we make computers do it). 

In the end we would have a likelihood of the observation for each value of $\mu$, we then would weight this value by our prior belief, so in the case of our prior guess being  that $\mu$ equals 3 with probability 25% and the likelihoood of 42% , we get that the probability that $\mu$ being equal to 3, given the fact that we observed 4 wins is $25\% * 42\% * C$ where $C$ is a constant we have to calculate as well, but is not needed to be understood here (it is about 0.25). So we have that the probability that $\mu$ is equal to 3, given the observed 4 wins is 25%. The probability of $\mu=4$ is now 20%, so we can see that now we believe that our prior beliefs have changed quite a lot (we used to think that $\mu=4$ has probability 5%).

This was probably very confusing and my apologies are in order

## The actual model
I am writing this half to explain it, and half to figure out exactly how exactly the model should work. Hence, the lack of clarity.

Although probably not as powerful as the almighty _algorithm_ by Jackson Krueger, I hope to be able to compare his ideas to mine. From now on we will use skill level and rating interchangebly. We will use a latent variable to measure the skill level of a given team $i$. We will call this $S_i$. This is the variable we are interested in measuring, but we will never see it directly. We will say that in a given game, each team realizes a skill level based on some distribution. The team with the higher realized skill level then wins the match. Hence, we have that 
$$ \mathbb{P} [ i \textrm{ wins against } j ] = \mathbb{P} [ S_i > S_j ] .$$
We have the problem of draws here (if the distributions are continuous), I have not figured out exactly how I am going to deal with that yet.

For simplicity, we will work with normal distributions for now. We will say each team's skill is distributed around a mean $mu_i$ with standard deviation $\sigma_i$ (this may end up being the same for all teams). Our prior is then on the $mu_i$ (may have to have one on the $\sigma_i$ as well if we do not fix those at the beginning), let this just be a standard normal for now. Then with a match between $i$ and $j$, we are observing the probability that $i>j$. We then get a posterior after the match (say $i$ wins)
$$\pi(\mu_i = x, \mu_j = y | i \textrm{ won}] = \frac{f(i>j | x,y) \pi(x,y)}{\int f(i>j | x,y) \pi(x,y) dxdy }.$$
Where $f(i>j | x,y)$ is the probability that the difference of independent normals centered around $x$ and $y$ is positive, i.e. 
$$\mathbb{P}\left[Z > - \frac{\mu_i - \mu_j}{\sigma_i + \sigma_j} \right].$$
This is not looking like it is going to have a nice conjugate prior. But that is an effort for later. So for now this will have to be done numerically.

To do this numerically each team will start out with a standard normally distributed random variable, this is our original prior. After observing a result, we shall be able to update this, with the observation (say team $i$ wins) conditional on the skill difference being normally distributed about the difference. The probability to observe $i$ winning will the be the area under the distribution to the right of 0 (if the difference is $S_i - S_j$, otherwise to the left). The parameters used for the numerical simulation are that each distribution will be represented by 1000 points between $[-5,5]$ (step size 0.01). 
