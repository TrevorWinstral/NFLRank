# MathNFL-PowerRank
I have never done something like this. I would like to apply various quantitative approaches to better understand NFL results and create power rankings. My first ideas are
- Applying Elo to measure skill level of teams.
- A Bayesian approach to latent skill variable.

Below I explain each of these approaches, the results will be below in a table.

Eventually I would like to evaluate these results against betting lines and my own thoughts. I will try and record what I think the win probability for each game is every week.

All data for matchups comes from [Pro Football Reference](https://www.pro-football-reference.com/years/2022/games.htm).

---
## Elo
The information on Elo rating system comes from the respective [Wikipedia page](https://en.wikipedia.org/wiki/Elo_rating_system). Here each team $i$ will have a rating $R_i$. I follow the Elo suggested scaling, namely that a difference of 200 points in 2 teams ratings should yield an expected score of 0.75, this means that the team has a 75% chance of winning (if we ignore draws). In reality we will account for draws with a draw being worth 0.5 wins. Hence the expected score for team $j$ when playing against team $j$ will be
$$E_i = \frac{1}{1 + 10^{(R_j - R_i)/400}}.$$
The rank update will occur after each week and works as follows. A team $i$ shall either achieve 1, 0.5, or 0 points based on realized results each week (by winning, drawing, or losing respectively, denoted by $S_i$. The difference between this score and their expected score will be multiplied by a factor $K$, finally this adjustment will be added to their original score.
$$R_i^{\textrm{new}} = R_i^{\textrm{old}} + K (S_i - E_i).$$
$K$ represents the maximal possible adjustment per game, we will experiment with different values here, chess has used 16 for masters and 32 for noobs.
Each team will start out with a ranking of 1000, and each week their ranking will be updated to tell us where they stand. Furthermore a predicted winner of each match will be calculated.

Note that we do not take much into account here, things such as home field advantage are not looked after at all. At week 1, each team will have a equal probability to win or lose, so it may take a few weeks for the ranks to establish themselves. To address this, I will start the simulation from the 2017/18 season. We will reset the ranks using the rank reset factor chosen by optimization.

Another factor I wanted to address was to account for was how much a team won by. In week 3 the Giants 19-16 win over the Panthers should not be valued the same as the Bills 41-7 absolute decimation of the Titans. To do this a win will not count as a score of 1, instead we will use a logistic function based of the point differential. I will fit this logistic function such that the 3 point win is worth around 0.65 points, a 7 point win around 0.8, and a 10 point win around 0.9 points. The 41-7 beatdown will now be worth 0.999 points for the winning team. A logistic factor $\beta=-0.1$ fulfills these conditions, and yields a score curve which looks like:

![Score Curve](./score_curve.svg)

In implementing the model a range a values were tested across the 2020/21 and 2021/22 season to minimize the mean square error between the expected and realized scores. The values which were minimized over were the logistic rate, the maximal rank change $K$, and a 'reset factor' $r$ which shrinks the ranks after each season. So a reset factor of $0.1$ means that after each season, a team only keep 10% of their previous rank. Testing a range of values for these 3 variables yielded a minimal mean square error for a logistic rate of 100, $K=11$, and $r=0.68$. This yields the following power ranking for week 3 of 2022/23 (read left to right, top to bottom):

<table border="1" class="dataframe">
  <tbody>
    <tr>
      <td>Kansas City Chiefs</td>
      <td>Tampa Bay Buccaneers</td>
      <td>Buffalo Bills</td>
      <td>Los Angeles Rams</td>
    </tr>
    <tr>
      <td>Green Bay Packers</td>
      <td>Pittsburgh Steelers</td>
      <td>Miami Dolphins</td>
      <td>New Orleans Saints</td>
    </tr>
    <tr>
      <td>Tennessee Titans</td>
      <td>San Francisco 49ers</td>
      <td>Minnesota Vikings</td>
      <td>Cincinnati Bengals</td>
    </tr>
    <tr>
      <td>Cleveland Browns</td>
      <td>Los Angeles Chargers</td>
      <td>Dallas Cowboys</td>
      <td>Indianapolis Colts</td>
    </tr>
    <tr>
      <td>Seattle Seahawks</td>
      <td>Baltimore Ravens</td>
      <td>Philadelphia Eagles</td>
      <td>Las Vegas Raiders</td>
    </tr>
    <tr>
      <td>New England Patriots</td>
      <td>Arizona Cardinals</td>
      <td>Chicago Bears</td>
      <td>Washington Commanders</td>
    </tr>
    <tr>
      <td>Atlanta Falcons</td>
      <td>New York Giants</td>
      <td>Denver Broncos</td>
      <td>Houston Texans</td>
    </tr>
    <tr>
      <td>Detroit Lions</td>
      <td>Carolina Panthers</td>
      <td>New York Jets</td>
      <td>Jacksonville Jaguars</td>
    </tr>
  </tbody>
</table>

And gave the Rams a 52% chance to win in the season opener against the Bills (not a great start, but much better than the experts on the pregame show who went around 70% for the Rams). I will continue to update the Elo power rankings each week, hopefully get that automated here soon, as well as some graphics to display the evolution over time.


### Matchup Predictions
The following table lists the matchups, and the predicted chance for team 1 to win (0.46 corresponds to 46%). This is for week 3 of 2022/23.
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Team 1</th>
      <th>Team 2</th>
      <th>Prediction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Pittsburgh Steelers</td>
      <td>Cleveland Browns</td>
      <td>0.563489</td>
    </tr>
    <tr>
      <td>Buffalo Bills</td>
      <td>Miami Dolphins</td>
      <td>0.567731</td>
    </tr>
    <tr>
      <td>Cincinnati Bengals</td>
      <td>New York Jets</td>
      <td>0.654492</td>
    </tr>
    <tr>
      <td>Detroit Lions</td>
      <td>Minnesota Vikings</td>
      <td>0.366806</td>
    </tr>
    <tr>
      <td>Houston Texans</td>
      <td>Chicago Bears</td>
      <td>0.390765</td>
    </tr>
    <tr>
      <td>Kansas City Chiefs</td>
      <td>Indianapolis Colts</td>
      <td>0.729253</td>
    </tr>
    <tr>
      <td>New Orleans Saints</td>
      <td>Carolina Panthers</td>
      <td>0.732336</td>
    </tr>
    <tr>
      <td>Philadelphia Eagles</td>
      <td>Washington Commanders</td>
      <td>0.613094</td>
    </tr>
    <tr>
      <td>Las Vegas Raiders</td>
      <td>Tennessee Titans</td>
      <td>0.443129</td>
    </tr>
    <tr>
      <td>Baltimore Ravens</td>
      <td>New England Patriots</td>
      <td>0.452316</td>
    </tr>
    <tr>
      <td>Jacksonville Jaguars</td>
      <td>Los Angeles Chargers</td>
      <td>0.311022</td>
    </tr>
    <tr>
      <td>Atlanta Falcons</td>
      <td>Seattle Seahawks</td>
      <td>0.393221</td>
    </tr>
    <tr>
      <td>Green Bay Packers</td>
      <td>Tampa Bay Buccaneers</td>
      <td>0.428180</td>
    </tr>
    <tr>
      <td>Los Angeles Rams</td>
      <td>Arizona Cardinals</td>
      <td>0.646301</td>
    </tr>
    <tr>
      <td>San Francisco 49ers</td>
      <td>Denver Broncos</td>
      <td>0.658135</td>
    </tr>
    <tr>
      <td>Dallas Cowboys</td>
      <td>New York Giants</td>
      <td>0.635412</td>
    </tr>
  </tbody>
</table>

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
$$\mathbb{P} [ i \textrm{ wins against } j ] = \mathbb{P} [ S_i > S_j ] .$$
We have the problem of draws here (if the distributions are continuous), I have not figured out exactly how I am going to deal with that yet.

For simplicity, we will work with normal distributions for now. We will say each team's skill is distributed around a mean $mu_i$ with standard deviation $\sigma_i$ (this may end up being the same for all teams). Our prior is then on the $mu_i$ (may have to have one on the $\sigma_i$ as well if we do not fix those at the beginning), let this just be a standard normal for now. Then with a match between $i$ and $j$, we are observing the probability that $i>j$. We then get a posterior after the match (say $i$ wins)
$$\pi(\mu_i = x, \mu_j = y | i \textrm{ won}] = \frac{f(i>j | x,y) \pi(x,y)}{\int f(i>j | x,y) \pi(x,y) dxdy }.$$
Where $f(i>j | x,y)$ is the probability that the difference of independent normals centered around $x$ and $y$ is positive, i.e. 
$$\mathbb{P}\left[Z > - \frac{\mu_i - \mu_j}{\sigma_i + \sigma_j} \right].$$
This is not looking like it is going to have a nice conjugate prior. But that is an effort for later. So for now this will have to be done numerically.

To do this numerically each team will start out with a standard normally distributed random variable, this is our original prior. After observing a result, we shall be able to update this, with the observation (say team $i$ wins) conditional on the skill difference being normally distributed about the difference. The probability to observe $i$ winning will the be the area under the distribution to the right of 0 (if the difference is $S_i - S_j$, otherwise to the left). The parameters used for the numerical simulation are that each distribution will be represented by 1000 points between $[-5,5]$ (step size 0.01). 


# Historical ratings
## Elo Ratings
Here are the historical ratings generated by the Elo algorithm. The column name corresponds to the rankings _after_ that week. The most recent ranking is on the far left, the oldest on the far right. Week 0 corresponds to the rankings after the last season which were used to initialize this year.

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Week 2</th>
      <th>Week 1</th>
      <th>Week 0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Kansas City Chiefs</td>
      <td>Kansas City Chiefs</td>
      <td>Los Angeles Rams</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tampa Bay Buccaneers</td>
      <td>Tampa Bay Buccaneers</td>
      <td>Kansas City Chiefs</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Buffalo Bills</td>
      <td>Buffalo Bills</td>
      <td>Tampa Bay Buccaneers</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Los Angeles Rams</td>
      <td>Los Angeles Rams</td>
      <td>Green Bay Packers</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Green Bay Packers</td>
      <td>Green Bay Packers</td>
      <td>Buffalo Bills</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Miami Dolphins</td>
      <td>Pittsburgh Steelers</td>
      <td>Tennessee Titans</td>
    </tr>
    <tr>
      <th>6</th>
      <td>San Francisco 49ers</td>
      <td>Miami Dolphins</td>
      <td>San Francisco 49ers</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Dallas Cowboys</td>
      <td>New Orleans Saints</td>
      <td>Cincinnati Bengals</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Pittsburgh Steelers</td>
      <td>San Francisco 49ers</td>
      <td>Pittsburgh Steelers</td>
    </tr>
    <tr>
      <th>9</th>
      <td>New Orleans Saints</td>
      <td>Tennessee Titans</td>
      <td>Dallas Cowboys</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Philadelphia Eagles</td>
      <td>Cincinnati Bengals</td>
      <td>Miami Dolphins</td>
    </tr>
    <tr>
      <th>11</th>
      <td>New England Patriots</td>
      <td>Minnesota Vikings</td>
      <td>New Orleans Saints</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Tennessee Titans</td>
      <td>Cleveland Browns</td>
      <td>Indianapolis Colts</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Arizona Cardinals</td>
      <td>Los Angeles Chargers</td>
      <td>Las Vegas Raiders</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Cincinnati Bengals</td>
      <td>Dallas Cowboys</td>
      <td>New England Patriots</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Minnesota Vikings</td>
      <td>Indianapolis Colts</td>
      <td>Cleveland Browns</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Los Angeles Chargers</td>
      <td>Seattle Seahawks</td>
      <td>Arizona Cardinals</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Seattle Seahawks</td>
      <td>Baltimore Ravens</td>
      <td>Minnesota Vikings</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Baltimore Ravens</td>
      <td>Philadelphia Eagles</td>
      <td>Los Angeles Chargers</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Cleveland Browns</td>
      <td>Las Vegas Raiders</td>
      <td>Seattle Seahawks</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Indianapolis Colts</td>
      <td>New England Patriots</td>
      <td>Baltimore Ravens</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Las Vegas Raiders</td>
      <td>Arizona Cardinals</td>
      <td>Philadelphia Eagles</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Chicago Bears</td>
      <td>Chicago Bears</td>
      <td>Washington Commanders</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Washington Commanders</td>
      <td>Washington Commanders</td>
      <td>Chicago Bears</td>
    </tr>
    <tr>
      <th>24</th>
      <td>New York Giants</td>
      <td>Atlanta Falcons</td>
      <td>Atlanta Falcons</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Denver Broncos</td>
      <td>New York Giants</td>
      <td>Denver Broncos</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Atlanta Falcons</td>
      <td>Denver Broncos</td>
      <td>Houston Texans</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Detroit Lions</td>
      <td>Houston Texans</td>
      <td>Detroit Lions</td>
    </tr>
    <tr>
      <th>28</th>
      <td>New York Jets</td>
      <td>Detroit Lions</td>
      <td>New York Giants</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Houston Texans</td>
      <td>Carolina Panthers</td>
      <td>Carolina Panthers</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Jacksonville Jaguars</td>
      <td>New York Jets</td>
      <td>New York Jets</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Carolina Panthers</td>
      <td>Jacksonville Jaguars</td>
      <td>Jacksonville Jaguars</td>
    </tr>
  </tbody>
</table>

## Elo Predictions
Here is the performance of the Elo Algorithm. The color denotes if the correct winner was chosen (green=correct winner, red=wrong), the opacity denotes how correct (strong green= almost perfect, strong red= completely wrong).
<style type="text/css">
#T_13c13_row0_col0, #T_13c13_row0_col1, #T_13c13_row0_col2, #T_13c13_row0_col3, #T_13c13_row0_col4, #T_13c13_row0_col5, #T_13c13_row0_col6, #T_13c13_row0_col7 {
  background_color: rgba(255,0,0,0.511);
}
#T_13c13_row1_col0, #T_13c13_row1_col1, #T_13c13_row1_col2, #T_13c13_row1_col3, #T_13c13_row1_col4, #T_13c13_row1_col5, #T_13c13_row1_col6, #T_13c13_row1_col7 {
  background_color: rgba(0,255,0,0.394);
}
#T_13c13_row2_col0, #T_13c13_row2_col1, #T_13c13_row2_col2, #T_13c13_row2_col3, #T_13c13_row2_col4, #T_13c13_row2_col5, #T_13c13_row2_col6, #T_13c13_row2_col7 {
  background_color: rgba(0,255,0,0.227);
}
#T_13c13_row3_col0, #T_13c13_row3_col1, #T_13c13_row3_col2, #T_13c13_row3_col3, #T_13c13_row3_col4, #T_13c13_row3_col5, #T_13c13_row3_col6, #T_13c13_row3_col7 {
  background_color: rgba(0,255,0,0.079);
}
#T_13c13_row4_col0, #T_13c13_row4_col1, #T_13c13_row4_col2, #T_13c13_row4_col3, #T_13c13_row4_col4, #T_13c13_row4_col5, #T_13c13_row4_col6, #T_13c13_row4_col7 {
  background_color: rgba(0,255,0,0.259);
}
#T_13c13_row5_col0, #T_13c13_row5_col1, #T_13c13_row5_col2, #T_13c13_row5_col3, #T_13c13_row5_col4, #T_13c13_row5_col5, #T_13c13_row5_col6, #T_13c13_row5_col7 {
  background_color: rgba(0,255,0,0.251);
}
#T_13c13_row6_col0, #T_13c13_row6_col1, #T_13c13_row6_col2, #T_13c13_row6_col3, #T_13c13_row6_col4, #T_13c13_row6_col5, #T_13c13_row6_col6, #T_13c13_row6_col7 {
  background_color: rgba(0,255,0,0.02);
}
#T_13c13_row7_col0, #T_13c13_row7_col1, #T_13c13_row7_col2, #T_13c13_row7_col3, #T_13c13_row7_col4, #T_13c13_row7_col5, #T_13c13_row7_col6, #T_13c13_row7_col7 {
  background_color: rgba(0,255,0,0.449);
}
#T_13c13_row8_col0, #T_13c13_row8_col1, #T_13c13_row8_col2, #T_13c13_row8_col3, #T_13c13_row8_col4, #T_13c13_row8_col5, #T_13c13_row8_col6, #T_13c13_row8_col7 {
  background_color: rgba(255,0,0,0.7989999999999999);
}
#T_13c13_row9_col0, #T_13c13_row9_col1, #T_13c13_row9_col2, #T_13c13_row9_col3, #T_13c13_row9_col4, #T_13c13_row9_col5, #T_13c13_row9_col6, #T_13c13_row9_col7 {
  background_color: rgba(255,0,0,0.73);
}
#T_13c13_row10_col0, #T_13c13_row10_col1, #T_13c13_row10_col2, #T_13c13_row10_col3, #T_13c13_row10_col4, #T_13c13_row10_col5, #T_13c13_row10_col6, #T_13c13_row10_col7 {
  background_color: rgba(255,0,0,0.20199999999999996);
}
#T_13c13_row11_col0, #T_13c13_row11_col1, #T_13c13_row11_col2, #T_13c13_row11_col3, #T_13c13_row11_col4, #T_13c13_row11_col5, #T_13c13_row11_col6, #T_13c13_row11_col7 {
  background_color: rgba(0,255,0,0.1);
}
#T_13c13_row12_col0, #T_13c13_row12_col1, #T_13c13_row12_col2, #T_13c13_row12_col3, #T_13c13_row12_col4, #T_13c13_row12_col5, #T_13c13_row12_col6, #T_13c13_row12_col7 {
  background_color: rgba(0,255,0,0.305);
}
#T_13c13_row13_col0, #T_13c13_row13_col1, #T_13c13_row13_col2, #T_13c13_row13_col3, #T_13c13_row13_col4, #T_13c13_row13_col5, #T_13c13_row13_col6, #T_13c13_row13_col7 {
  background_color: rgba(0,255,0,0.171);
}
#T_13c13_row14_col0, #T_13c13_row14_col1, #T_13c13_row14_col2, #T_13c13_row14_col3, #T_13c13_row14_col4, #T_13c13_row14_col5, #T_13c13_row14_col6, #T_13c13_row14_col7 {
  background_color: rgba(255,0,0,0.5489999999999999);
}
#T_13c13_row15_col0, #T_13c13_row15_col1, #T_13c13_row15_col2, #T_13c13_row15_col3, #T_13c13_row15_col4, #T_13c13_row15_col5, #T_13c13_row15_col6, #T_13c13_row15_col7 {
  background_color: rgba(0,255,0,0.022);
}
#T_13c13_row16_col0, #T_13c13_row16_col1, #T_13c13_row16_col2, #T_13c13_row16_col3, #T_13c13_row16_col4, #T_13c13_row16_col5, #T_13c13_row16_col6, #T_13c13_row16_col7 {
  background_color: rgba(0,255,0,0.036);
}
#T_13c13_row17_col0, #T_13c13_row17_col1, #T_13c13_row17_col2, #T_13c13_row17_col3, #T_13c13_row17_col4, #T_13c13_row17_col5, #T_13c13_row17_col6, #T_13c13_row17_col7 {
  background_color: rgba(0,255,0,0.405);
}
#T_13c13_row18_col0, #T_13c13_row18_col1, #T_13c13_row18_col2, #T_13c13_row18_col3, #T_13c13_row18_col4, #T_13c13_row18_col5, #T_13c13_row18_col6, #T_13c13_row18_col7 {
  background_color: rgba(255,0,0,0.31200000000000006);
}
#T_13c13_row19_col0, #T_13c13_row19_col1, #T_13c13_row19_col2, #T_13c13_row19_col3, #T_13c13_row19_col4, #T_13c13_row19_col5, #T_13c13_row19_col6, #T_13c13_row19_col7 {
  background_color: rgba(0,255,0,0.353);
}
#T_13c13_row20_col0, #T_13c13_row20_col1, #T_13c13_row20_col2, #T_13c13_row20_col3, #T_13c13_row20_col4, #T_13c13_row20_col5, #T_13c13_row20_col6, #T_13c13_row20_col7 {
  background_color: rgba(0,255,0,0.217);
}
#T_13c13_row21_col0, #T_13c13_row21_col1, #T_13c13_row21_col2, #T_13c13_row21_col3, #T_13c13_row21_col4, #T_13c13_row21_col5, #T_13c13_row21_col6, #T_13c13_row21_col7 {
  background_color: rgba(255,0,0,0.6819999999999999);
}
#T_13c13_row22_col0, #T_13c13_row22_col1, #T_13c13_row22_col2, #T_13c13_row22_col3, #T_13c13_row22_col4, #T_13c13_row22_col5, #T_13c13_row22_col6, #T_13c13_row22_col7 {
  background_color: rgba(0,255,0,0.016);
}
#T_13c13_row23_col0, #T_13c13_row23_col1, #T_13c13_row23_col2, #T_13c13_row23_col3, #T_13c13_row23_col4, #T_13c13_row23_col5, #T_13c13_row23_col6, #T_13c13_row23_col7 {
  background_color: rgba(255,0,0,0.475);
}
#T_13c13_row24_col0, #T_13c13_row24_col1, #T_13c13_row24_col2, #T_13c13_row24_col3, #T_13c13_row24_col4, #T_13c13_row24_col5, #T_13c13_row24_col6, #T_13c13_row24_col7 {
  background_color: rgba(255,0,0,0.46299999999999997);
}
#T_13c13_row25_col0, #T_13c13_row25_col1, #T_13c13_row25_col2, #T_13c13_row25_col3, #T_13c13_row25_col4, #T_13c13_row25_col5, #T_13c13_row25_col6, #T_13c13_row25_col7 {
  background_color: rgba(0,255,0,0.228);
}
#T_13c13_row26_col0, #T_13c13_row26_col1, #T_13c13_row26_col2, #T_13c13_row26_col3, #T_13c13_row26_col4, #T_13c13_row26_col5, #T_13c13_row26_col6, #T_13c13_row26_col7 {
  background_color: rgba(255,0,0,0.765);
}
#T_13c13_row27_col0, #T_13c13_row27_col1, #T_13c13_row27_col2, #T_13c13_row27_col3, #T_13c13_row27_col4, #T_13c13_row27_col5, #T_13c13_row27_col6, #T_13c13_row27_col7 {
  background_color: rgba(255,0,0,0.732);
}
#T_13c13_row28_col0, #T_13c13_row28_col1, #T_13c13_row28_col2, #T_13c13_row28_col3, #T_13c13_row28_col4, #T_13c13_row28_col5, #T_13c13_row28_col6, #T_13c13_row28_col7 {
  background_color: rgba(0,255,0,0.097);
}
#T_13c13_row29_col0, #T_13c13_row29_col1, #T_13c13_row29_col2, #T_13c13_row29_col3, #T_13c13_row29_col4, #T_13c13_row29_col5, #T_13c13_row29_col6, #T_13c13_row29_col7 {
  background_color: rgba(0,255,0,0.052);
}
#T_13c13_row30_col0, #T_13c13_row30_col1, #T_13c13_row30_col2, #T_13c13_row30_col3, #T_13c13_row30_col4, #T_13c13_row30_col5, #T_13c13_row30_col6, #T_13c13_row30_col7 {
  background_color: rgba(0,255,0,0.109);
}
#T_13c13_row31_col0, #T_13c13_row31_col1, #T_13c13_row31_col2, #T_13c13_row31_col3, #T_13c13_row31_col4, #T_13c13_row31_col5, #T_13c13_row31_col6, #T_13c13_row31_col7 {
  background_color: rgba(0,255,0,0.44);
}
</style>
<table id="T_13c13">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_13c13_level0_col0" class="col_heading level0 col0" >Date</th>
      <th id="T_13c13_level0_col1" class="col_heading level0 col1" >Winner/tie</th>
      <th id="T_13c13_level0_col2" class="col_heading level0 col2" >Loser/tie</th>
      <th id="T_13c13_level0_col3" class="col_heading level0 col3" >Game Score</th>
      <th id="T_13c13_level0_col4" class="col_heading level0 col4" >Normalized Score</th>
      <th id="T_13c13_level0_col5" class="col_heading level0 col5" >Prediction</th>
      <th id="T_13c13_level0_col6" class="col_heading level0 col6" >Qualitative Error</th>
      <th id="T_13c13_level0_col7" class="col_heading level0 col7" >Error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_13c13_level0_row0" class="row_heading level0 row0" >31</th>
      <td id="T_13c13_row0_col0" class="data row0 col0" >2022-09-19</td>
      <td id="T_13c13_row0_col1" class="data row0 col1" >Philadelphia Eagles</td>
      <td id="T_13c13_row0_col2" class="data row0 col2" >Minnesota Vikings</td>
      <td id="T_13c13_row0_col3" class="data row0 col3" >24-7</td>
      <td id="T_13c13_row0_col4" class="data row0 col4" >0.980000</td>
      <td id="T_13c13_row0_col5" class="data row0 col5" >0.491000</td>
      <td id="T_13c13_row0_col6" class="data row0 col6" >False</td>
      <td id="T_13c13_row0_col7" class="data row0 col7" >0.489000</td>
    </tr>
    <tr>
      <th id="T_13c13_level0_row1" class="row_heading level0 row1" >30</th>
      <td id="T_13c13_row1_col0" class="data row1 col0" >2022-09-19</td>
      <td id="T_13c13_row1_col1" class="data row1 col1" >Buffalo Bills</td>
      <td id="T_13c13_row1_col2" class="data row1 col2" >Tennessee Titans</td>
      <td id="T_13c13_row1_col3" class="data row1 col3" >41-7</td>
      <td id="T_13c13_row1_col4" class="data row1 col4" >1.000000</td>
      <td id="T_13c13_row1_col5" class="data row1 col5" >0.606000</td>
      <td id="T_13c13_row1_col6" class="data row1 col6" >True</td>
      <td id="T_13c13_row1_col7" class="data row1 col7" >0.394000</td>
    </tr>
    <tr>
      <th id="T_13c13_level0_row2" class="row_heading level0 row2" >29</th>
      <td id="T_13c13_row2_col0" class="data row2 col0" >2022-09-18</td>
      <td id="T_13c13_row2_col1" class="data row2 col1" >Green Bay Packers</td>
      <td id="T_13c13_row2_col2" class="data row2 col2" >Chicago Bears</td>
      <td id="T_13c13_row2_col3" class="data row2 col3" >27-10</td>
      <td id="T_13c13_row2_col4" class="data row2 col4" >0.980000</td>
      <td id="T_13c13_row2_col5" class="data row2 col5" >0.754000</td>
      <td id="T_13c13_row2_col6" class="data row2 col6" >True</td>
      <td id="T_13c13_row2_col7" class="data row2 col7" >0.227000</td>
    </tr>
    <tr>
      <th id="T_13c13_level0_row3" class="row_heading level0 row3" >28</th>
      <td id="T_13c13_row3_col0" class="data row3 col0" >2022-09-18</td>
      <td id="T_13c13_row3_col1" class="data row3 col1" >Dallas Cowboys</td>
      <td id="T_13c13_row3_col2" class="data row3 col2" >Cincinnati Bengals</td>
      <td id="T_13c13_row3_col3" class="data row3 col3" >20-17</td>
      <td id="T_13c13_row3_col4" class="data row3 col4" >0.666000</td>
      <td id="T_13c13_row3_col5" class="data row3 col5" >0.588000</td>
      <td id="T_13c13_row3_col6" class="data row3 col6" >True</td>
      <td id="T_13c13_row3_col7" class="data row3 col7" >0.079000</td>
    </tr>
    <tr>
      <th id="T_13c13_level0_row4" class="row_heading level0 row4" >27</th>
      <td id="T_13c13_row4_col0" class="data row4 col0" >2022-09-18</td>
      <td id="T_13c13_row4_col1" class="data row4 col1" >Arizona Cardinals</td>
      <td id="T_13c13_row4_col2" class="data row4 col2" >Las Vegas Raiders</td>
      <td id="T_13c13_row4_col3" class="data row4 col3" >29-23</td>
      <td id="T_13c13_row4_col4" class="data row4 col4" >0.799000</td>
      <td id="T_13c13_row4_col5" class="data row4 col5" >0.540000</td>
      <td id="T_13c13_row4_col6" class="data row4 col6" >True</td>
      <td id="T_13c13_row4_col7" class="data row4 col7" >0.259000</td>
    </tr>
    <tr>
      <th id="T_13c13_level0_row5" class="row_heading level0 row5" >26</th>
      <td id="T_13c13_row5_col0" class="data row5 col0" >2022-09-18</td>
      <td id="T_13c13_row5_col1" class="data row5 col1" >Denver Broncos</td>
      <td id="T_13c13_row5_col2" class="data row5 col2" >Houston Texans</td>
      <td id="T_13c13_row5_col3" class="data row5 col3" >16-9</td>
      <td id="T_13c13_row5_col4" class="data row5 col4" >0.834000</td>
      <td id="T_13c13_row5_col5" class="data row5 col5" >0.583000</td>
      <td id="T_13c13_row5_col6" class="data row5 col6" >True</td>
      <td id="T_13c13_row5_col7" class="data row5 col7" >0.251000</td>
    </tr>
    <tr>
      <th id="T_13c13_level0_row6" class="row_heading level0 row6" >25</th>
      <td id="T_13c13_row6_col0" class="data row6 col0" >2022-09-18</td>
      <td id="T_13c13_row6_col1" class="data row6 col1" >Los Angeles Rams</td>
      <td id="T_13c13_row6_col2" class="data row6 col2" >Atlanta Falcons</td>
      <td id="T_13c13_row6_col3" class="data row6 col3" >31-27</td>
      <td id="T_13c13_row6_col4" class="data row6 col4" >0.715000</td>
      <td id="T_13c13_row6_col5" class="data row6 col5" >0.736000</td>
      <td id="T_13c13_row6_col6" class="data row6 col6" >True</td>
      <td id="T_13c13_row6_col7" class="data row6 col7" >-0.020000</td>
    </tr>
    <tr>
      <th id="T_13c13_level0_row7" class="row_heading level0 row7" >24</th>
      <td id="T_13c13_row7_col0" class="data row7 col0" >2022-09-18</td>
      <td id="T_13c13_row7_col1" class="data row7 col1" >San Francisco 49ers</td>
      <td id="T_13c13_row7_col2" class="data row7 col2" >Seattle Seahawks</td>
      <td id="T_13c13_row7_col3" class="data row7 col3" >27-7</td>
      <td id="T_13c13_row7_col4" class="data row7 col4" >0.990000</td>
      <td id="T_13c13_row7_col5" class="data row7 col5" >0.541000</td>
      <td id="T_13c13_row7_col6" class="data row7 col6" >True</td>
      <td id="T_13c13_row7_col7" class="data row7 col7" >0.449000</td>
    </tr>
    <tr>
      <th id="T_13c13_level0_row8" class="row_heading level0 row8" >23</th>
      <td id="T_13c13_row8_col0" class="data row8 col0" >2022-09-18</td>
      <td id="T_13c13_row8_col1" class="data row8 col1" >New York Giants</td>
      <td id="T_13c13_row8_col2" class="data row8 col2" >Carolina Panthers</td>
      <td id="T_13c13_row8_col3" class="data row8 col3" >19-16</td>
      <td id="T_13c13_row8_col4" class="data row8 col4" >0.666000</td>
      <td id="T_13c13_row8_col5" class="data row8 col5" >0.465000</td>
      <td id="T_13c13_row8_col6" class="data row8 col6" >False</td>
      <td id="T_13c13_row8_col7" class="data row8 col7" >0.201000</td>
    </tr>
    <tr>
      <th id="T_13c13_level0_row9" class="row_heading level0 row9" >22</th>
      <td id="T_13c13_row9_col0" class="data row9 col0" >2022-09-18</td>
      <td id="T_13c13_row9_col1" class="data row9 col1" >New York Jets</td>
      <td id="T_13c13_row9_col2" class="data row9 col2" >Cleveland Browns</td>
      <td id="T_13c13_row9_col3" class="data row9 col3" >31-30</td>
      <td id="T_13c13_row9_col4" class="data row9 col4" >0.557000</td>
      <td id="T_13c13_row9_col5" class="data row9 col5" >0.287000</td>
      <td id="T_13c13_row9_col6" class="data row9 col6" >False</td>
      <td id="T_13c13_row9_col7" class="data row9 col7" >0.270000</td>
    </tr>
    <tr>
      <th id="T_13c13_level0_row10" class="row_heading level0 row10" >21</th>
      <td id="T_13c13_row10_col0" class="data row10 col0" >2022-09-18</td>
      <td id="T_13c13_row10_col1" class="data row10 col1" >Jacksonville Jaguars</td>
      <td id="T_13c13_row10_col2" class="data row10 col2" >Indianapolis Colts</td>
      <td id="T_13c13_row10_col3" class="data row10 col3" >24-0</td>
      <td id="T_13c13_row10_col4" class="data row10 col4" >0.996000</td>
      <td id="T_13c13_row10_col5" class="data row10 col5" >0.198000</td>
      <td id="T_13c13_row10_col6" class="data row10 col6" >False</td>
      <td id="T_13c13_row10_col7" class="data row10 col7" >0.798000</td>
    </tr>
    <tr>
      <th id="T_13c13_level0_row11" class="row_heading level0 row11" >20</th>
      <td id="T_13c13_row11_col0" class="data row11 col0" >2022-09-18</td>
      <td id="T_13c13_row11_col1" class="data row11 col1" >New England Patriots</td>
      <td id="T_13c13_row11_col2" class="data row11 col2" >Pittsburgh Steelers</td>
      <td id="T_13c13_row11_col3" class="data row11 col3" >17-14</td>
      <td id="T_13c13_row11_col4" class="data row11 col4" >0.666000</td>
      <td id="T_13c13_row11_col5" class="data row11 col5" >0.567000</td>
      <td id="T_13c13_row11_col6" class="data row11 col6" >True</td>
      <td id="T_13c13_row11_col7" class="data row11 col7" >0.100000</td>
    </tr>
    <tr>
      <th id="T_13c13_level0_row12" class="row_heading level0 row12" >19</th>
      <td id="T_13c13_row12_col0" class="data row12 col0" >2022-09-18</td>
      <td id="T_13c13_row12_col1" class="data row12 col1" >Tampa Bay Buccaneers</td>
      <td id="T_13c13_row12_col2" class="data row12 col2" >New Orleans Saints</td>
      <td id="T_13c13_row12_col3" class="data row12 col3" >20-10</td>
      <td id="T_13c13_row12_col4" class="data row12 col4" >0.909000</td>
      <td id="T_13c13_row12_col5" class="data row12 col5" >0.604000</td>
      <td id="T_13c13_row12_col6" class="data row12 col6" >True</td>
      <td id="T_13c13_row12_col7" class="data row12 col7" >0.305000</td>
    </tr>
    <tr>
      <th id="T_13c13_level0_row13" class="row_heading level0 row13" >18</th>
      <td id="T_13c13_row13_col0" class="data row13 col0" >2022-09-18</td>
      <td id="T_13c13_row13_col1" class="data row13 col1" >Miami Dolphins</td>
      <td id="T_13c13_row13_col2" class="data row13 col2" >Baltimore Ravens</td>
      <td id="T_13c13_row13_col3" class="data row13 col3" >42-38</td>
      <td id="T_13c13_row13_col4" class="data row13 col4" >0.715000</td>
      <td id="T_13c13_row13_col5" class="data row13 col5" >0.545000</td>
      <td id="T_13c13_row13_col6" class="data row13 col6" >True</td>
      <td id="T_13c13_row13_col7" class="data row13 col7" >0.171000</td>
    </tr>
    <tr>
      <th id="T_13c13_level0_row14" class="row_heading level0 row14" >17</th>
      <td id="T_13c13_row14_col0" class="data row14 col0" >2022-09-18</td>
      <td id="T_13c13_row14_col1" class="data row14 col1" >Detroit Lions</td>
      <td id="T_13c13_row14_col2" class="data row14 col2" >Washington Commanders</td>
      <td id="T_13c13_row14_col3" class="data row14 col3" >36-27</td>
      <td id="T_13c13_row14_col4" class="data row14 col4" >0.888000</td>
      <td id="T_13c13_row14_col5" class="data row14 col5" >0.437000</td>
      <td id="T_13c13_row14_col6" class="data row14 col6" >False</td>
      <td id="T_13c13_row14_col7" class="data row14 col7" >0.451000</td>
    </tr>
    <tr>
      <th id="T_13c13_level0_row15" class="row_heading level0 row15" >16</th>
      <td id="T_13c13_row15_col0" class="data row15 col0" >2022-09-15</td>
      <td id="T_13c13_row15_col1" class="data row15 col1" >Kansas City Chiefs</td>
      <td id="T_13c13_row15_col2" class="data row15 col2" >Los Angeles Chargers</td>
      <td id="T_13c13_row15_col3" class="data row15 col3" >27-24</td>
      <td id="T_13c13_row15_col4" class="data row15 col4" >0.666000</td>
      <td id="T_13c13_row15_col5" class="data row15 col5" >0.688000</td>
      <td id="T_13c13_row15_col6" class="data row15 col6" >True</td>
      <td id="T_13c13_row15_col7" class="data row15 col7" >-0.022000</td>
    </tr>
    <tr>
      <th id="T_13c13_level0_row16" class="row_heading level0 row16" >15</th>
      <td id="T_13c13_row16_col0" class="data row16 col0" >2022-09-12</td>
      <td id="T_13c13_row16_col1" class="data row16 col1" >Seattle Seahawks</td>
      <td id="T_13c13_row16_col2" class="data row16 col2" >Denver Broncos</td>
      <td id="T_13c13_row16_col3" class="data row16 col3" >17-16</td>
      <td id="T_13c13_row16_col4" class="data row16 col4" >0.557000</td>
      <td id="T_13c13_row16_col5" class="data row16 col5" >0.593000</td>
      <td id="T_13c13_row16_col6" class="data row16 col6" >True</td>
      <td id="T_13c13_row16_col7" class="data row16 col7" >-0.036000</td>
    </tr>
    <tr>
      <th id="T_13c13_level0_row17" class="row_heading level0 row17" >14</th>
      <td id="T_13c13_row17_col0" class="data row17 col0" >2022-09-11</td>
      <td id="T_13c13_row17_col1" class="data row17 col1" >Tampa Bay Buccaneers</td>
      <td id="T_13c13_row17_col2" class="data row17 col2" >Dallas Cowboys</td>
      <td id="T_13c13_row17_col3" class="data row17 col3" >19-3</td>
      <td id="T_13c13_row17_col4" class="data row17 col4" >0.975000</td>
      <td id="T_13c13_row17_col5" class="data row17 col5" >0.570000</td>
      <td id="T_13c13_row17_col6" class="data row17 col6" >True</td>
      <td id="T_13c13_row17_col7" class="data row17 col7" >0.405000</td>
    </tr>
    <tr>
      <th id="T_13c13_level0_row18" class="row_heading level0 row18" >13</th>
      <td id="T_13c13_row18_col0" class="data row18 col0" >2022-09-11</td>
      <td id="T_13c13_row18_col1" class="data row18 col1" >Minnesota Vikings</td>
      <td id="T_13c13_row18_col2" class="data row18 col2" >Green Bay Packers</td>
      <td id="T_13c13_row18_col3" class="data row18 col3" >23-7</td>
      <td id="T_13c13_row18_col4" class="data row18 col4" >0.975000</td>
      <td id="T_13c13_row18_col5" class="data row18 col5" >0.288000</td>
      <td id="T_13c13_row18_col6" class="data row18 col6" >False</td>
      <td id="T_13c13_row18_col7" class="data row18 col7" >0.688000</td>
    </tr>
    <tr>
      <th id="T_13c13_level0_row19" class="row_heading level0 row19" >12</th>
      <td id="T_13c13_row19_col0" class="data row19 col0" >2022-09-11</td>
      <td id="T_13c13_row19_col1" class="data row19 col1" >Kansas City Chiefs</td>
      <td id="T_13c13_row19_col2" class="data row19 col2" >Arizona Cardinals</td>
      <td id="T_13c13_row19_col3" class="data row19 col3" >44-21</td>
      <td id="T_13c13_row19_col4" class="data row19 col4" >0.995000</td>
      <td id="T_13c13_row19_col5" class="data row19 col5" >0.642000</td>
      <td id="T_13c13_row19_col6" class="data row19 col6" >True</td>
      <td id="T_13c13_row19_col7" class="data row19 col7" >0.353000</td>
    </tr>
    <tr>
      <th id="T_13c13_level0_row20" class="row_heading level0 row20" >11</th>
      <td id="T_13c13_row20_col0" class="data row20 col0" >2022-09-11</td>
      <td id="T_13c13_row20_col1" class="data row20 col1" >Los Angeles Chargers</td>
      <td id="T_13c13_row20_col2" class="data row20 col2" >Las Vegas Raiders</td>
      <td id="T_13c13_row20_col3" class="data row20 col3" >24-19</td>
      <td id="T_13c13_row20_col4" class="data row20 col4" >0.760000</td>
      <td id="T_13c13_row20_col5" class="data row20 col5" >0.543000</td>
      <td id="T_13c13_row20_col6" class="data row20 col6" >True</td>
      <td id="T_13c13_row20_col7" class="data row20 col7" >0.217000</td>
    </tr>
    <tr>
      <th id="T_13c13_level0_row21" class="row_heading level0 row21" >10</th>
      <td id="T_13c13_row21_col0" class="data row21 col0" >2022-09-11</td>
      <td id="T_13c13_row21_col1" class="data row21 col1" >New York Giants</td>
      <td id="T_13c13_row21_col2" class="data row21 col2" >Tennessee Titans</td>
      <td id="T_13c13_row21_col3" class="data row21 col3" >21-20</td>
      <td id="T_13c13_row21_col4" class="data row21 col4" >0.557000</td>
      <td id="T_13c13_row21_col5" class="data row21 col5" >0.239000</td>
      <td id="T_13c13_row21_col6" class="data row21 col6" >False</td>
      <td id="T_13c13_row21_col7" class="data row21 col7" >0.318000</td>
    </tr>
    <tr>
      <th id="T_13c13_level0_row22" class="row_heading level0 row22" >9</th>
      <td id="T_13c13_row22_col0" class="data row22 col0" >2022-09-11</td>
      <td id="T_13c13_row22_col1" class="data row22 col1" >Cleveland Browns</td>
      <td id="T_13c13_row22_col2" class="data row22 col2" >Carolina Panthers</td>
      <td id="T_13c13_row22_col3" class="data row22 col3" >26-24</td>
      <td id="T_13c13_row22_col4" class="data row22 col4" >0.613000</td>
      <td id="T_13c13_row22_col5" class="data row22 col5" >0.629000</td>
      <td id="T_13c13_row22_col6" class="data row22 col6" >True</td>
      <td id="T_13c13_row22_col7" class="data row22 col7" >-0.016000</td>
    </tr>
    <tr>
      <th id="T_13c13_level0_row23" class="row_heading level0 row23" >8</th>
      <td id="T_13c13_row23_col0" class="data row23 col0" >2022-09-11</td>
      <td id="T_13c13_row23_col1" class="data row23 col1" >Miami Dolphins</td>
      <td id="T_13c13_row23_col2" class="data row23 col2" >New England Patriots</td>
      <td id="T_13c13_row23_col3" class="data row23 col3" >20-7</td>
      <td id="T_13c13_row23_col4" class="data row23 col4" >0.952000</td>
      <td id="T_13c13_row23_col5" class="data row23 col5" >0.428000</td>
      <td id="T_13c13_row23_col6" class="data row23 col6" >False</td>
      <td id="T_13c13_row23_col7" class="data row23 col7" >0.525000</td>
    </tr>
    <tr>
      <th id="T_13c13_level0_row24" class="row_heading level0 row24" >7</th>
      <td id="T_13c13_row24_col0" class="data row24 col0" >2022-09-11</td>
      <td id="T_13c13_row24_col1" class="data row24 col1" >Chicago Bears</td>
      <td id="T_13c13_row24_col2" class="data row24 col2" >San Francisco 49ers</td>
      <td id="T_13c13_row24_col3" class="data row24 col3" >19-10</td>
      <td id="T_13c13_row24_col4" class="data row24 col4" >0.888000</td>
      <td id="T_13c13_row24_col5" class="data row24 col5" >0.351000</td>
      <td id="T_13c13_row24_col6" class="data row24 col6" >False</td>
      <td id="T_13c13_row24_col7" class="data row24 col7" >0.537000</td>
    </tr>
    <tr>
      <th id="T_13c13_level0_row25" class="row_heading level0 row25" >6</th>
      <td id="T_13c13_row25_col0" class="data row25 col0" >2022-09-11</td>
      <td id="T_13c13_row25_col1" class="data row25 col1" >Baltimore Ravens</td>
      <td id="T_13c13_row25_col2" class="data row25 col2" >New York Jets</td>
      <td id="T_13c13_row25_col3" class="data row25 col3" >24-9</td>
      <td id="T_13c13_row25_col4" class="data row25 col4" >0.969000</td>
      <td id="T_13c13_row25_col5" class="data row25 col5" >0.741000</td>
      <td id="T_13c13_row25_col6" class="data row25 col6" >True</td>
      <td id="T_13c13_row25_col7" class="data row25 col7" >0.228000</td>
    </tr>
    <tr>
      <th id="T_13c13_level0_row26" class="row_heading level0 row26" >5</th>
      <td id="T_13c13_row26_col0" class="data row26 col0" >2022-09-11</td>
      <td id="T_13c13_row26_col1" class="data row26 col1" >Pittsburgh Steelers</td>
      <td id="T_13c13_row26_col2" class="data row26 col2" >Cincinnati Bengals</td>
      <td id="T_13c13_row26_col3" class="data row26 col3" >23-20</td>
      <td id="T_13c13_row26_col4" class="data row26 col4" >0.666000</td>
      <td id="T_13c13_row26_col5" class="data row26 col5" >0.431000</td>
      <td id="T_13c13_row26_col6" class="data row26 col6" >False</td>
      <td id="T_13c13_row26_col7" class="data row26 col7" >0.235000</td>
    </tr>
    <tr>
      <th id="T_13c13_level0_row27" class="row_heading level0 row27" >4</th>
      <td id="T_13c13_row27_col0" class="data row27 col0" >2022-09-11</td>
      <td id="T_13c13_row27_col1" class="data row27 col1" >Houston Texans</td>
      <td id="T_13c13_row27_col2" class="data row27 col2" >Indianapolis Colts</td>
      <td id="T_13c13_row27_col3" class="data row27 col3" >20-20</td>
      <td id="T_13c13_row27_col4" class="data row27 col4" >0.500000</td>
      <td id="T_13c13_row27_col5" class="data row27 col5" >0.232000</td>
      <td id="T_13c13_row27_col6" class="data row27 col6" >False</td>
      <td id="T_13c13_row27_col7" class="data row27 col7" >0.268000</td>
    </tr>
    <tr>
      <th id="T_13c13_level0_row28" class="row_heading level0 row28" >3</th>
      <td id="T_13c13_row28_col0" class="data row28 col0" >2022-09-11</td>
      <td id="T_13c13_row28_col1" class="data row28 col1" >Washington Commanders</td>
      <td id="T_13c13_row28_col2" class="data row28 col2" >Jacksonville Jaguars</td>
      <td id="T_13c13_row28_col3" class="data row28 col3" >28-22</td>
      <td id="T_13c13_row28_col4" class="data row28 col4" >0.799000</td>
      <td id="T_13c13_row28_col5" class="data row28 col5" >0.703000</td>
      <td id="T_13c13_row28_col6" class="data row28 col6" >True</td>
      <td id="T_13c13_row28_col7" class="data row28 col7" >0.097000</td>
    </tr>
    <tr>
      <th id="T_13c13_level0_row29" class="row_heading level0 row29" >2</th>
      <td id="T_13c13_row29_col0" class="data row29 col0" >2022-09-11</td>
      <td id="T_13c13_row29_col1" class="data row29 col1" >Philadelphia Eagles</td>
      <td id="T_13c13_row29_col2" class="data row29 col2" >Detroit Lions</td>
      <td id="T_13c13_row29_col3" class="data row29 col3" >38-35</td>
      <td id="T_13c13_row29_col4" class="data row29 col4" >0.666000</td>
      <td id="T_13c13_row29_col5" class="data row29 col5" >0.718000</td>
      <td id="T_13c13_row29_col6" class="data row29 col6" >True</td>
      <td id="T_13c13_row29_col7" class="data row29 col7" >-0.052000</td>
    </tr>
    <tr>
      <th id="T_13c13_level0_row30" class="row_heading level0 row30" >1</th>
      <td id="T_13c13_row30_col0" class="data row30 col0" >2022-09-11</td>
      <td id="T_13c13_row30_col1" class="data row30 col1" >New Orleans Saints</td>
      <td id="T_13c13_row30_col2" class="data row30 col2" >Atlanta Falcons</td>
      <td id="T_13c13_row30_col3" class="data row30 col3" >27-26</td>
      <td id="T_13c13_row30_col4" class="data row30 col4" >0.557000</td>
      <td id="T_13c13_row30_col5" class="data row30 col5" >0.667000</td>
      <td id="T_13c13_row30_col6" class="data row30 col6" >True</td>
      <td id="T_13c13_row30_col7" class="data row30 col7" >-0.109000</td>
    </tr>
    <tr>
      <th id="T_13c13_level0_row31" class="row_heading level0 row31" >0</th>
      <td id="T_13c13_row31_col0" class="data row31 col0" >2022-09-08</td>
      <td id="T_13c13_row31_col1" class="data row31 col1" >Buffalo Bills</td>
      <td id="T_13c13_row31_col2" class="data row31 col2" >Los Angeles Rams</td>
      <td id="T_13c13_row31_col3" class="data row31 col3" >31-10</td>
      <td id="T_13c13_row31_col4" class="data row31 col4" >0.992000</td>
      <td id="T_13c13_row31_col5" class="data row31 col5" >0.552000</td>
      <td id="T_13c13_row31_col6" class="data row31 col6" >True</td>
      <td id="T_13c13_row31_col7" class="data row31 col7" >0.440000</td>
    </tr>
  </tbody>
</table>
