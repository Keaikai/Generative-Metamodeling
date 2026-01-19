library(simmer)

#setwd("D:/QRGMM/Bank_Simulator")

dir_list <- c(
  "./data",
  "./data/traindata",
  "./data/testdata",
  "./data/onlinetest",
  "./data/testdata_QRNNGMM",
  "./data/onlinetest/testdata_QRNNGMM_online"
    
)

for (d in dir_list) {
  if (!dir.exists(d)) {
    dir.create(d, recursive = TRUE)
  }
}

bank_simulator <- function(lambda1, lambda2, service_rate1, service_rate2, service_rate3, 
                           service_rate4, service_rate5, service_rate6, m1, m2) {
  service_rates <- list(
    counter1 = service_rate1,
    counter2 = service_rate2,
    counter3 = service_rate3,
    counter4 = service_rate4,
    counter5 = service_rate5,
    counter6 = service_rate6
  )
  
  open_time <- 9 * 60  # 9:00 AM in minutes
  close_time <- 17 * 60  # 5:00 PM in minutes
  maxTime <- (17 + 4) * 60  # Maximum simulation time in minutes
  
  bank <- simmer()
  
  normal_customer <-
    trajectory("Customer's path") %>%
    set_attribute("start_time", function() {now(bank)}) %>%
    renege_in(function() {rgamma(1, shape = m1, scale = 1)},
              out = trajectory("Reneging customer") ) %>%
    select(c("counter1", "counter2", "counter3", "counter4", "counter5", "counter6"), policy = "shortest-queue") %>%
    seize_selected() %>%
    renege_abort() %>%
    timeout(function() {
      selected_counter <- get_selected(bank)
      rexp(1, service_rates[[selected_counter]])
    }) %>%
    release_selected() 
  
  priority_customer <-
    trajectory("Customer's path") %>%
    set_attribute("start_time", function() {now(bank)}) %>%
    renege_in(function() {rgamma(1, shape = m2, scale = 1)},
              out = trajectory("Reneging customer") ) %>%
    select(c("counter1", "counter2", "counter3", "counter4", "counter5", "counter6"), policy = "shortest-queue") %>%
    seize_selected() %>%
    renege_abort() %>%
    timeout(function() {
      selected_counter <- get_selected(bank)
      rexp(1, service_rates[[selected_counter]])
    }) %>%
    release_selected() 
  
  bank <-
    simmer("bank") %>%
    add_resource("counter1", 1) %>%
    add_resource("counter2", 1) %>%
    add_resource("counter3", 1) %>%
    add_resource("counter4", 1) %>%
    add_resource("counter5", 1) %>%
    add_resource("counter6", 1) %>%
    add_generator("NormalCustomer", normal_customer, from_to(open_time, close_time, function() {rexp(1, lambda1)})) %>%
    add_generator("PriorityCustomer", priority_customer, from_to(open_time, close_time, function() {rexp(1, lambda2)}), priority = 1)
  
  bank %>% run(until = maxTime)
  
  # Retrieve the results and store them in the arrivals dataframe
  arrivals <- get_mon_arrivals(bank) %>%
    transform(system_time = end_time - start_time)
  
  # Filter NormalCustomer and PriorityCustomer using grepl
  normal_customers <- arrivals[grepl("NormalCustomer", arrivals$name), ]
  priority_customers <- arrivals[grepl("PriorityCustomer", arrivals$name), ]
  
  # Calculate the renege count and renege rate for NormalCustomer
  number_reneged_normal <- sum(normal_customers$activity_time == 0)
  renege_rate_normal <- number_reneged_normal / nrow(normal_customers)
  
  # Calculate the renege count and renege rate for PriorityCustomer
  number_reneged_priority <- sum(priority_customers$activity_time == 0)
  renege_rate_priority <- number_reneged_priority / nrow(priority_customers)
  
  # Calculate the average system time for NormalCustomer
  average_system_time_normal <- mean(normal_customers$system_time)
  
  # Calculate the average system time for PriorityCustomer
  average_system_time_priority <- mean(priority_customers$system_time)
  
  # Calculate maximum overtime
  max_overtime <- max(arrivals$end_time) - close_time
  max_overtime <- ifelse(max_overtime > 0, max_overtime, 0)  # Ensure overtime is not negative
  
  # Return the results as a list
  return(list(
    average_system_time_normal = average_system_time_normal,
    renege_rate_normal = renege_rate_normal,
    average_system_time_priority = average_system_time_priority,
    renege_rate_priority = renege_rate_priority,
    max_overtime = max_overtime
  ))
  
}


# Prepare the train data and test data

# Define the input ranges for the 10-dimensional vector X
lambda1_range <- c(1/5, 1)
lambda2_range <- c(1/10, 1/5)

service_rate_range <- list(
  counter1 = c(1/20, 1/10),
  counter2 = c(1/20, 1/10),
  counter3 = c(1/20, 1/10),
  counter4 = c(1/20, 1/10),
  counter5 = c(1/20, 1/10),
  counter6 = c(1/20, 1/10)
)

m1_range <- c(15, 20)
m2_range <- c(10, 15)

# Function to sample within a given range
sample_uniform <- function(range) {
  runif(1, min = range[1], max = range[2])
}

# Function to run the simulation for a given input vector X
simulate_bank <- function(X) {
  results <- bank_simulator(
    lambda1 = X[1], lambda2 = X[2], 
    service_rate1 = X[3], service_rate2 = X[4], 
    service_rate3 = X[5], service_rate4 = X[6], 
    service_rate5 = X[7], service_rate6 = X[8], 
    m1 = X[9], m2 = X[10]
  )
  return(results)
}

n_samples <- 10000

n_rep <- 100

for (rep in 1:n_rep) {
  
  set.seed(rep)  # Use the replicate number as the seed
  
  # Sample 10,000 input vectors x
  X_train<-matrix(0, nrow = n_samples, ncol = 10)
  X_train[,1] <- replicate(n_samples, sample_uniform(lambda1_range))
  X_train[,2] <- replicate(n_samples, sample_uniform(lambda2_range))
  X_train[,3] <- replicate(n_samples, sample_uniform(service_rate_range$counter1))
  X_train[,4] <- replicate(n_samples, sample_uniform(service_rate_range$counter2))
  X_train[,5] <- replicate(n_samples, sample_uniform(service_rate_range$counter3))
  X_train[,6] <- replicate(n_samples, sample_uniform(service_rate_range$counter4))
  X_train[,7] <- replicate(n_samples, sample_uniform(service_rate_range$counter5))
  X_train[,8] <- replicate(n_samples, sample_uniform(service_rate_range$counter6))
  X_train[,9] <- replicate(n_samples, sample_uniform(m1_range))
  X_train[,10] <- replicate(n_samples, sample_uniform(m2_range))

  # Run simulation for each x and store the results
  train_data <- matrix(0, nrow = nrow(X_train), ncol = 15)
  train_data[,1:10]=X_train
  for (i in 1:nrow(X_train)) {
    result <- simulate_bank(X_train[i, ])
    train_data[i, 11:15] <- c(
      result$average_system_time_normal,
      result$renege_rate_normal,
      result$average_system_time_priority,
      result$renege_rate_priority,
      result$max_overtime
    )
  }
  path1 <- paste("./data/traindata/train_data_rep",rep,sep="")
  path1 <- paste(path1,".csv",sep="")
  write.csv(train_data,path1,row.names = FALSE)

  
  # Generate test data by uniformly sampling 10,000 new points of x and simulating each once
  X_test <- matrix(0, nrow = n_samples, ncol = 10)
  X_test[,1] <- replicate(n_samples, sample_uniform(lambda1_range))
  X_test[,2] <- replicate(n_samples, sample_uniform(lambda2_range))
  X_test[,3] <- replicate(n_samples, sample_uniform(service_rate_range$counter1))
  X_test[,4] <- replicate(n_samples, sample_uniform(service_rate_range$counter2))
  X_test[,5] <- replicate(n_samples, sample_uniform(service_rate_range$counter3))
  X_test[,6] <- replicate(n_samples, sample_uniform(service_rate_range$counter4))
  X_test[,7] <- replicate(n_samples, sample_uniform(service_rate_range$counter5))
  X_test[,8] <- replicate(n_samples, sample_uniform(service_rate_range$counter6))
  X_test[,9] <- replicate(n_samples, sample_uniform(m1_range))
  X_test[,10] <- replicate(n_samples, sample_uniform(m2_range))
  
  # Run simulation for each x and store the results
  test_data <- matrix(0, nrow = nrow(X_test), ncol = 15)
  test_data[,1:10]=X_test
  for (i in 1:nrow(X_test)) {
    result <- simulate_bank(X_test[i, ])
    test_data[i, 11:15] <- c(
      result$average_system_time_normal,
      result$renege_rate_normal,
      result$average_system_time_priority,
      result$renege_rate_priority,
      result$max_overtime
    )
  }
  path2 <- paste("./data/testdata/test_data_rep",rep,sep="")
  path2 <- paste(path2,".csv",sep="")
  write.csv(test_data,path2,row.names = FALSE)
  

}


set.seed(2024) 
K <- 10000
# Select a specific point x_0 for a new test set
x_0 <-c(1/4,1/8,1/15,1/16,1/17,1/19,1/11,1/13,18,12)
X_0 <- matrix(rep(x_0, K), nrow = K, ncol = 10, byrow = TRUE)
# Simulate x_0 10,000 times to create a new test data
test_data_x0 <- matrix(0, nrow = K, ncol = 15)
test_data_x0[,1:10]=X_0


for (i in 1:K) {
  result <- simulate_bank(X_0[i, ])
  test_data_x0[i, 11:15] <- c(
    result$average_system_time_normal,
    result$renege_rate_normal,
    result$average_system_time_priority,
    result$renege_rate_priority,
    result$max_overtime
  )
}

path3 <- paste("./data/onlinetest/test_data_x0.csv")
write.csv(test_data_x0,path3,row.names = FALSE)




# # An example for running the bank simulator
# 
# set.seed(999)
# results <- bank_simulator(lambda1 = 1/8, lambda2 = 1/12, 
#                           service_rate1 = 1/25, service_rate2 = 1/26, 
#                           service_rate3 = 1/21, service_rate4 = 1/30, 
#                           service_rate5 = 1/24, service_rate6 = 1/28, 
#                           m1 = 30, m2 = 20)
# print(results)