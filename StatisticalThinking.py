################################################################################################
#Introduction to Exploratory Data Analysis
################################################################################################
################################################
#Plotting histogram
################################################
#Method 1 - matplotlib
import matplotlib.pyplot as plt
plt.hist(df_swing["dem_share"],bins=20)
plt.show()
#Method2 - Seaborn
import seaborn as sns
sns.set()# Set default Seaborn style
plt.hist(versicolor_petal_length)
plt.ylabel("count")
plt.xlabel("petal length (cm)")
plt.show()

## Adjusting the number of bins in a histogram
import numpy as np
n_data = len(versicolor_petal_length) #number of data points
n_bins = np.sqrt(n_data)
n_bins = int(n_bins)# Convert number of bins to integer
# Plot the histogram
_ = plt.hist(versicolor_petal_length, bins=n_bins)
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('count')
plt.show()
################################################
#Bee swarm plot
################################################
# bee swarm plot with Seaborn's default settings
sns.swarmplot(x="species",y="petal length (cm)",data=df)
plt.xlabel("species")
plt.ylabel("petal length (cm)")
plt.show()

################################################
#Empirical cumulative distribution function (ECDF)
################################################
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1) / n
    return x, y

## Plotting
# Compute ECDF for versicolor data
x_vers, y_vers = ecdf(versicolor_petal_length)
plt.plot(x_vers,y_vers,marker='.', linestyle = 'none')
plt.ylabel("EDCF")
plt.xlabel("x")
plt.show()

## Comparison EDCF
# Compute ECDFs
x_set, y_set = ecdf(setosa_petal_length)
x_vers, y_vers = ecdf(versicolor_petal_length)
x_virg, y_virg = ecdf(virginica_petal_length)
# Plot all ECDFs on the same plot
_ = plt.plot(x_set, y_set, marker='.', linestyle='none')
_ = plt.plot(x_vers, y_vers, marker='.', linestyle='none')
_ = plt.plot(x_virg, y_virg, marker='.', linestyle='none')
# Annotate the plot
_ = plt.legend(('setosa', 'versicolor', 'virginica'), loc='lower right')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')
plt.show()
################################################################################################
#Introduction to summary statistics: The sample mean and median
################################################################################################
## Compute the mean: 
mean_length_vers = np.mean(versicolor_petal_length)
print('I. versicolor:', mean_length_vers, 'cm')

## Compute percentiles: 
percentiles=np.array([2.5,25,50,75,97.5])
ptiles_vers = np.percentile(versicolor_petal_length,percentiles)
print(ptiles_vers)

## Plot the ECDF
_ = plt.plot(x_vers, y_vers, '.')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')
# Overlay percentiles as red diamonds.
_ = plt.plot(ptiles_vers, percentiles/100, marker='D', color='red',
         linestyle='none')
plt.show()

## Box-and-whisker plot
_ =sns.boxplot(x='species',y='petal length (cm)',data=df)
plt.xlabel("species")
plt.ylabel("petal length (cm)")
plt.show()

## Computing the variance
# Array of differences to mean: 
differences = versicolor_petal_length - np.mean(versicolor_petal_length)
# Square the differences: 
diff_sq=(differences)**2
# Compute the mean square difference: 
variance_explicit = np.mean(diff_sq)
# Compute the variance using NumPy
variance_np = np.var(versicolor_petal_length)
print(variance_explicit,variance_np)

## The standard deviation and the variance
# variance:
variance = np.var(versicolor_petal_length)
# square root of the variance
print(np.sqrt(variance))
# standard deviation
print(np.std(versicolor_petal_length))
################################################
#Covariance and the Pearson correlation coefficient
################################################
## Make a scatter plot
plt.plot(versicolor_petal_length,versicolor_petal_width,marker='.',linestyle='none')
plt.xlabel("versicolor_petal_length")
plt.ylabel("versicolor_petal_width")
plt.show()

## Compute the covariance matrix: 
covariance_matrix = np.cov(versicolor_petal_length,versicolor_petal_width)
print(covariance_matrix)

## Extract covariance of length and width of petals: 
petal_cov=covariance_matrix[0,1]
print(petal_cov)

## Computing the Pearson correlation coefficient
def pearson_r(x, y):
    # Compute correlation matrix: 
    corr_mat = np.corrcoef(x,y)
    # Return entry [0,1]
    return corr_mat[0,1]
r=pearson_r(versicolor_petal_length , versicolor_petal_width)
print(r)
################################################################################################
#Thinking probabilistically-- Discrete variables
#Probabilistic logic and statistical inference
    #Probability provides a measure of uncertainty.
    #Data are almost never exactly the same when acquired again, and probability allows us to say how much we expect them to vary.
################################################################################################
## Random number generators and hacker statistics
    #Hacker statistics
        #-Uses simulated repeated measurements to compute probabilities.
    #The np.random module
        #Suite of functions based on random number generation
        #np.random.random(): draw a number between 0 and 1
################################################
np.random.seed(42)# Seed the random number generator
random_numbers = np.empty(100000)# Initialize random numbers: random_numbers
# Generate random numbers by looping over range(100000)
for i in range(100000):
    random_numbers[i] = np.random.random()
_ = plt.hist(random_numbers)# Plot a histogram
plt.show()

## The np.random module and Bernoulli trials
def perform_bernoulli_trials(n, p):
    # Initialize number of successes: n_success
    n_success = 0
    # Perform trials
    for i in range(n):
        # Choose random number between zero and one: random_number
        random_number = np.random.random()
        # If less than p, it's a success  so add one to n_success
        if random_number < p:
            n_success += 1
    return n_success

## How many defaults might we expect?
# Seed random number generator
np.random.seed(42)
# Initialize the number of defaults: 
n_defaults = np.empty(1000)
# Compute the number of defaults
for i in range(1000):
    n_defaults[i] = perform_bernoulli_trials(100,0.05)
# Plot the histogram with default number of bins; label your axes
_ = plt.hist(n_defaults, normed=True)
_ = plt.xlabel('number of defaults out of 100 loans')
_ = plt.ylabel('probability')
plt.show()

## Will the bank fail?
# Compute ECDF: 
x, y = ecdf(n_defaults)
plt.plot(x,y,marker = '.' , linestyle = 'none')
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Compute the number of 100-loan simulations with 10 or more defaults: 
n_lose_money = np.sum(n_defaults >= 10)
# Compute and print probability of losing money
print('Probability of losing money =', n_lose_money / len(n_defaults))
'''As we might expect, we most likely get 5/100 defaults. 
But we still have about a 2% chance of getting 10 or more defaults out of 100 loans'''
################################################
## Probability distributions and stories: The Binomial distribution
# Binomial distribution: the story
'''
The number r of successes in n Bernoulli trials with probability p of success, is Binomially distributed
The number r of heads in 4 coin flips with probability 0.5 of heads, is Binomially distributed'''
################################################
## Sampling out of the Binomial distribution
# Take 10,000 samples out of the binomial distribution: 
n_defaults = np.random.binomial(n = 100 , p = 0.05, size=10000)
# Compute CDF 
x, y = ecdf(n_defaults)
# Plot 
plt.plot(x,y,marker='.',linestyle='none')
plt.xlabel("x")
plt.ylabel("y")
plt.show()

## Plotting the Binomial PMF
# Compute bin edges: bins
bins = np.arange(max(n_defaults) + 1.5) - 0.5
plt.hist(n_defaults,normed=True,bins=bins)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
################################################
#Poisson processes and the Poisson distribution
The timing of the next event is completely
independent of when the previous event happened

Examples of Poisson processes
Natural births in a given hospital
Hit on a website during a given hour
Meteor strikes
Molecular collisions in a gas
Aviation incidents
Buses in Poissonville

Poisson distribution
The number r of arrivals of a Poisson process in a given time interval with average rate of ? arrivals per interval is Poisson distributed.
The number r of hits on a website in one hour with an average hit rate of 6 hits per hour is Poisson distributed.

Poisson Distribution
Limit of the Binomial distribution for low probability of success and large number of trials.
That is, for rare events.
################################################
## Relationship between Binomial and Poisson distributions
# Draw 10,000 samples out of Poisson distribution: 
samples_poisson=np.random.poisson(10,size=10000)
# Print the mean and standard deviation
print('Poisson:     ', np.mean(samples_poisson),
                       np.std(samples_poisson))

# Specify values of n and p to consider for Binomial: 
n = [20, 100, 1000] 
p = [0.5, 0.1, 0.01]
# Draw 10,000 samples for each n,p pair: samples_binomial
for i in range(3):
    samples_binomial = np.random.binomial(n[i],p[i],10000)
    print('n =', n[i], 'Binom:', np.mean(samples_binomial),
                                 np.std(samples_binomial))
################################################
## Was 2015 anomalous?
# Draw 10,000 samples out of Poisson distribution: 
n_nohitters = np.random.poisson(251/115, 10000)


# Compute number of samples that are seven or greater: n_large
n_large = np.sum(n_nohitters>=7)

# Compute probability of getting seven or more: 
p_large = n_large/10000
print('Probability of seven or more no-hitters:', p_large)

################################################################################################
#Probability density functions(PDF): relative likelihood of observing a value of a continuous variable
################################################################################################

################################################
#Introduction to the Normal distribution
#Normal distribution: PDF has a single symmetric peak
################################################
## The Normal PDF
# Draw 100000 samples from Normal distribution with stds of interest
mean = 20
samples_std1 = np.random.normal(mean,1, size=100000)
samples_std3= np.random.normal(mean,3, size=100000)
samples_std10= np.random.normal(mean,10, size=100000)
# Make histograms
plt.hist(samples_std1,bins=100,histtype='step',normed=True)
plt.hist(samples_std3,bins=100,histtype='step',normed=True)
plt.hist(samples_std10,bins=100,histtype='step',normed=True)
# Make a legend, set limits and show plot
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'))
plt.ylim(-0.01, 0.42)
plt.show()
################################################
## The Normal CDF
# Generate CDFs
x_std1, y_std1=ecdf(samples_std1)
x_std3, y_std3=ecdf(samples_std3)
x_std10, y_std10=ecdf(samples_std10)
# Plot CDFs
plt.plot(x_std1, y_std1,marker='.', linestyle='none') 
plt.plot(x_std3, y_std3,marker='.', linestyle='none') 
plt.plot(x_std10, y_std10,marker='.', linestyle='none') 
# Make a legend and show the plot
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'), loc='lower right')
plt.show()
################################################
#The Normal distribution: Properties and warnings
################################################
## The theoretical CDF VS the ECDF of the data: Belmont Stakes
# Compute mean and standard deviation: 
mu = np.mean(belmont_no_outliers)
sigma =np.std(belmont_no_outliers)
# Sample out of a normal distribution with this mu and sigma: 
samples = np.random.normal(mu,sigma,size=10000)
# Get the CDF of the samples and of the data
x_theor, y_theor=ecdf(samples)
x,y = ecdf(belmont_no_outliers)
# Plot the CDFs and show the plot
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('Belmont winning time (sec.)')
_ = plt.ylabel('CDF')
plt.show()
################################################
## What are the chances of a horse matching or beating Secretariat's record?
# Take a million samples out of the Normal distribution: 
samples = np.random.normal(mu, sigma,size=1000000)
# Compute the fraction that are faster than 144 seconds: 
prob = np.sum(samples <=144)/len(samples)
# Print the result
print('Probability of besting Secretariat:', prob)
################################################
#Exponential Distribution
 #-The Exponential distribution describes the waiting times between rare events, and Secretariat is rare!
 # expect the time between Major League no-hitters to be distributed? probability distribution of the time between no hitters.
################################################
def successive_poisson(tau1, tau2, size=1):
    """Compute time for arrival of 2 successive Poisson processes."""
    # Draw samples out of first exponential distribution: t1
    t1 = np.random.exponential(tau1, size=1)
    # Draw samples out of second exponential distribution: t2
    t2 = np.random.exponential(tau2, size=1)
    return t1 + t2
################################################
## Distribution of no-hitters and cycles
# Draw samples of waiting times
waiting_times = successive_poisson(764, 715, size=100000)
# Make the histogram
_ = plt.hist(waiting_times, bins=100, histtype='step',
             normed=True)
# Label axes
_ = plt.xlabel('total waiting time (games)')
_ = plt.ylabel('PDF')
################################################################################################
# Statistical Thinking II
'''
● Estimate parameters
● Compute confidence intervals
● Perform linear regressions
● Test hypotheses'''
################################################################################################
#Optimal parameters
################################################
## expoential distribution plot
np.random.seed(42)
tau = np.mean(nohitter_times)# Compute mean no-hitter time: tau
inter_nohitter_time = np.random.exponential(tau, 100000)# Draw out of an exponential distribution with parameter tau
_ = plt.hist(inter_nohitter_time,
             bins=50, normed=True, histtype='step')
_ = plt.xlabel('Games between no-hitters')
_ = plt.ylabel('PDF')
plt.show()
################################################
## real data VS theoretical model
# Create an ECDF from real data: x, y
x, y = ecdf(nohitter_times)
# Create a CDF from theoretical samples: x_theor, y_theor
x_theor, y_theor = ecdf(inter_nohitter_time)

# Overlay the plots
plt.plot(x_theor, y_theor)
plt.plot(x, y, marker='.', linestyle= 'none')
plt.margins(0.02)
plt.xlabel('Games between no-hitters')
plt.ylabel('CDF')
plt.show()
'''It looks like no-hitters in the modern era of Major League Baseball are Exponentially distributed.
Based on the story of the Exponential distribution, this suggests that they are a random process;
 when a no-hitter will happen is independent of when the last no-hitter was.'''
################################################
 ## optimal parameter?
# Plot the theoretical CDFs
plt.plot(x_theor, y_theor)
plt.plot(x, y, marker='.', linestyle='none')
plt.margins(0.02)
plt.xlabel('Games between no-hitters')
plt.ylabel('CDF')

# Take samples with half tau: samples_half
samples_half = np.random.exponential(tau/2, size=10000)
# Take samples with double tau: samples_double
samples_double = np.random.exponential(tau*2, size=10000)

# Generate CDFs from these samples
x_half, y_half = ecdf(samples_half)
x_double, y_double = ecdf(samples_double)
_ = plt.plot(x_half, y_half)
_ = plt.plot(x_double, y_double)
plt.show()
################################################
#Linear regression by least squares: The process of finding the parameters for which the sum of the squares of the residuals is minimal
################################################
## EDA of literacy/fertility data
# Plot the illiteracy rate versus fertility
_ = plt.plot(fertility, illiteracy, marker='.', linestyle='none')
plt.margins(0.02)
_ = plt.xlabel('percent illiterate')
_ = plt.ylabel('fertility')
plt.show()
################################################
## Linear regression
# linear regression 
a, b = np.polyfit(illiteracy,fertility,1)
print('slope =', a, 'children per woman / percent illiterate')
print('intercept =', b, 'children per woman')

# Make theoretical line to plot
x = np.array([0, 100])
y = a * x + b
# Add regression line to your plot
_ = plt.plot(x, y)
plt.show()

# Show the Pearson correlation coefficient
print(pearson_r(illiteracy, fertility))
################################################
## Optimal value?
# Specify slopes to consider: a_vals
a_vals = np.linspace(0,0.1,200) #200 points between 0 and 0.1
# Initialize sum of square of residuals: rss
rss = np.empty_like(a_vals)
# Compute sum of square of residuals for each value of a_vals
for i, a in enumerate(a_vals):
    rss[i] = np.sum((fertility - a*illiteracy - b)**2)
# Plot the RSS
plt.plot(a_vals, rss, '-')
plt.xlabel('slope (children per woman / percent illiterate)')
plt.ylabel('sum of square of residuals')
plt.show()
'''minimum on the plot, that is the value of the slope that gives the minimum sum of the square'''
################################################
# importance of EDA
################################################
## Linear regression on appropriate Anscombe data
# Perform linear regression: a, b
a, b = np.polyfit(x,y,1)
print(a, b)# Print the slope and intercept
# Generate theoretical x and y data
x_theor = np.array([3, 15])
y_theor = a * x_theor + b
_ = plt.plot(x,y, marker='.', linestyle='none')# Plot
_ = plt.plot(x_theor,y_theor)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

## Linear regression on all Anscombe data
for x, y in zip(anscombe_x, anscombe_y):
    # Compute the slope and intercept: a, b
    a, b = np.polyfit(x, y, 1)
    print('slope:', a, 'intercept:', b)
'''anscombe_x = [x1, x2, x3, x4] 
   anscombe_y = [y1, y2, y3, y4]'''
################################################
#Generating bootstrap replicates; resampling an array with replacement
# The use of resampled data to perform statistical inference
#  np.random.chocie()
################################################
## Visualizing bootstrap samples
for _ in range(50):
    # Generate bootstrap sample: bs_sample
    bs_sample = np.random.choice(rainfall, size=len(rainfall))

    # Compute and plot ECDF from bootstrap sample
    x, y = ecdf(bs_sample)
    _ = plt.plot(x, y, marker='.', linestyle='none',
                 color='gray', alpha=0.1)

    # Compute and plot ECDF from original data
    x, y = ecdf(rainfall)
    _ = plt.plot(x, y, marker='.')
    plt.margins(0.02)
    _ = plt.xlabel('yearly rainfall (mm)')
    _ = plt.ylabel('ECDF')
    plt.show()
################################################################################################
## Bootstrap confidence intervals
################################################################################################
## 1)Generating many bootstrap replicates
def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""
    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data,func)
    return bs_replicates

## 2)Bootstrap replicates of the mean and the SEM
# Take 10,000 bootstrap replicates of the mean: bs_replicates
bs_replicates = draw_bs_reps(rainfall,np.mean,10000)

# Compute and print SEM
sem = np.std(rainfall) / np.sqrt(len(rainfall))
print(sem)

# Compute and print standard deviation of bootstrap replicates
bs_std = np.std(bs_replicates)
print(bs_std)

# Make a histogram of the results
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel('mean annual rainfall (mm)')
_ = plt.ylabel('PDF')
plt.show()
''' SEM we got from the known expression and the bootstrap replicates is the same
and the distribution of the bootstrap replicates of the mean is Normal.'''

## 3)Confidence intervals of rainfall data
np.percentile(bs_replicates,[2.5,97.5])

## 4) Bootstrap replicates of other statistics
# Generate 10,000 bootstrap replicates of the variance
bs_replicates = draw_bs_reps(rainfall,np.var,10000)

# Put the variance in units of square centimeters
bs_replicates/100
# Make a histogram 
_ = plt.hist(bs_replicates/100, bins=50, normed=True)
_ = plt.xlabel('variance of annual rainfall (sq. cm)')
_ = plt.ylabel('PDF')
plt.show()
'''his is not normally distributed, as it has a longer tail to the right. '''
################################################
## Confidence interval on the rate of no-hitters
# Draw bootstrap replicates of the mean no-hitter time (equal to tau)
bs_replicates = draw_bs_reps(nohitter_times,np.mean,10000)
conf_int = np.percentile(bs_replicates,[2.5,97.5])# 95% confidence interval
print('95% confidence interval =', conf_int, 'games')
_ = plt.hist(bs_replicates, bins=50, normed=True)# Plot the histogram of the replicates
_ = plt.xlabel(r'$\tau$ (games)')
_ = plt.ylabel('PDF')
plt.show()
''' estimate of what the typical time between no-hitters is. It could be anywhere between 660 and 870 games.'''
################################################
#Pairs bootstrap
################################################
## 1) A function to do pairs bootstrap
def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y,1)

    return bs_slope_reps, bs_intercept_reps

## 2) Pairs bootstrap of literacy/fertility dat
# Generate replicates of slope and intercept using pairs bootstrap
bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(illiteracy,fertility,size=1000)

# Compute and print 95% CI for slope
print(np.percentile(bs_slope_reps, [2.5,97.5]))

# Plot the histogram
_ = plt.hist(bs_slope_reps, bins=50, normed=True)
_ = plt.xlabel('slope')
_ = plt.ylabel('PDF')
plt.show()

## 3) Plotting bootstrap regressions
# Generate array of x-values for bootstrap lines: x
x = np.array([0, 100])

# Plot the bootstrap lines
for i in range(100):
    _ = plt.plot(x, 
                 bs_slope_reps[i]*x + bs_intercept_reps[i],
                 linewidth=0.5, alpha=0.2, color='red')
# Plot the data
_ = plt.plot(illiteracy,fertility,marker='.', linestyle='none')
_ = plt.xlabel('illiteracy')
_ = plt.ylabel('fertility')
plt.margins(0.02)
plt.show()
################################################################################################
#Introduction to hypothesis testing
#Simulating the hypothesis
    #Permutation:Random reordering of entries in an array
################################################################################################

################################################