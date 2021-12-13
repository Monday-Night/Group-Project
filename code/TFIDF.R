library(foreign)
library(psych)
library(NLP) 
library(tm)
library(ldatuning)
library(slam)
library(lsa)
library(topicmodels)
library(ldatuning)
library(slam)
describe_text <- read.csv(file.choose(), header = TRUE, encoding="UTF-8", stringsAsFactors=FALSE)
names(describe_text)[names(describe_text) == 'X'] <- 'doc_id'
names(describe_text)[names(describe_text) == 'Describe'] <- 'text'
describe_text_docs <- subset(describe_text, select = c("doc_id", "text"))
describe_text_VCorpus <- VCorpus(DataframeSource(describe_text_docs)) 
describe_text_VCorpus <- tm_map(describe_text_VCorpus, content_transformer(tolower))
removeURL <- function(x) gsub("http[^[:space:]]*", "", x)
describe_text_VCorpus <- tm_map(describe_text_VCorpus, content_transformer(removeURL))
removeNumPunct <- function(x) gsub("[^[:alpha:][:space:]]*", "", x)
describe_text_VCorpus <- tm_map(describe_text_VCorpus, content_transformer(removeNumPunct))
describe_text_VCorpus <- tm_map(describe_text_VCorpus, removeWords, stopwords("english"))
describe_text_VCorpus <- tm_map(describe_text_VCorpus, stripWhitespace)
describe_text_VCorpus <- tm_map(describe_text_VCorpus, removePunctuation)
describe_text_dtm_tfidf <- DocumentTermMatrix(describe_text_VCorpus, control = list(weighting = weightTfIdf))
describe_text_dtm_tfidf2 = removeSparseTerms(describe_text_dtm_tfidf, 0.99)
write.csv(as.data.frame(sort(colSums(as.matrix(describe_text_dtm_tfidf2)), decreasing=TRUE)), file="describe_texT_TFIDF.csv")
describe_text_dtm <- DocumentTermMatrix(describe_text_VCorpus, control = list(removePunctuation = TRUE, stopwords=TRUE))
rowTotals_text <- apply(describe_text_dtm , 1, sum)
describe_text_dtm_nonzero <- describe_text_dtm[rowTotals_text> 0, ]
result <- FindTopicsNumber(
  describe_text_dtm_nonzero,
  topics = seq(from = 2, to = 15, by = 1),
  metrics = c("CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs",
  control = list(seed = 77),
  mc.cores = 2L,
  verbose = TRUE
)
FindTopicsNumber_plot(result)
describe_text_dtm_5topics <- LDA(describe_text_dtm_nonzero, k = 5, method = "Gibbs", control = list(iter=2000, seed = 2000))
describe_text_dtm_5topics_10words <- terms(describe_text_dtm_5topics, 10)
describe_text_dtm_5topics_10words