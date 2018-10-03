# Remove donor-specifically expressed genes 

library(tidyverse)

raw_data <- read_tsv('Brain_gene_exp.tsv') %>% as.data.frame()
donorID <- t(raw_data[2, -1])
gene_exp <- t(raw_data[-c(1:2), ])
colnames(gene_exp) <- gene_exp[1, ]
gene_exp <- gene_exp[-1, ]
data_with_donorID <- data.frame(donorID, gene_exp) 
data_with_donorID[, c(2:ncol(data_with_donorID))] <- sapply(data_with_donorID[, c(2:ncol(data_with_donorID))], as.numeric)
colnames(data_with_donorID)[1]<-'DonorID'

mean_exp <- data.frame(matrix(nrow = 2, ncol = ncol(data_with_donorID) - 1))
colnames(mean_exp) <- colnames(data_with_donorID)[-1]
for (i in 2:ncol(data_with_donorID)){
  mean_exp[1, i-1] <- mean(data_with_donorID[, i])
  mean_exp[2, i-1] <- sd(data_with_donorID[, i])
}

avg_by_donor <- aggregate(.~ DonorID, data_with_donorID, mean)
rownames(avg_by_donor) <- avg_by_donor[, 1]
avg_by_donor <- avg_by_donor[, -1]

gene_list <- c()
for (i in 1:ncol(avg_by_donor)){
  m <- mean_exp[1, i]
  sd <- mean_exp[2, i]
  if (max(avg_by_donor[, i]) > m + 3*sd | min(avg_by_donor[, i]) < m - 3*sd){
    gene_list <- c(gene_list, colnames(avg_by_donor)[i])
  }
}

raw_data$rm <- 0
for (i in 3:nrow(raw_data)){
  if (raw_data[i, 1] %in% gene_list){
    raw_data$rm[i] <- 1
  }
}

trimmed_data <- raw_data %>% filter(rm == 0)
trimmed_data <- trimmed_data[, -ncol(trimmed_data)]
write_tsv(trimmed_data, 'Brain_gene_exp.rm_donor_specific.tsv')
