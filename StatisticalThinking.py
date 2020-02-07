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

## Was 2015 anomalous?
# Draw 10,000 samples out of Poisson distribution: 
n_nohitters = np.random.poisson(251/115, 10000)


# Compute number of samples that are seven or greater: n_large
n_large = np.sum(n_nohitters>=7)

# Compute probability of getting seven or more: 
p_large = n_large/10000
print('Probability of seven or more no-hitters:', p_large)

################################################################################################
#Probability density functions
################################################################################################

################################################

################################################