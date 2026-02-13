from collections import defaultdict

# ----------------------------
# Training Corpus
# ----------------------------
corpus = [
    "<s> I love NLP </s>",
    "<s> I love deep learning </s>",
    "<s> deep learning is fun </s>"
]

# ----------------------------
# Count Unigrams and Bigrams
# ----------------------------
unigram_counts = defaultdict(int)
bigram_counts = defaultdict(int)

for sentence in corpus:
    words = sentence.split()
    
    # Count unigrams
    for word in words:
        unigram_counts[word] += 1
    
    # Count bigrams
    for i in range(1, len(words)):
        bigram = (words[i-1], words[i])
        bigram_counts[bigram] += 1

# ----------------------------
# Compute Bigram Probability (MLE)
# ----------------------------
def bigram_probability(prev_word, word):
    if unigram_counts[prev_word] == 0:
        return 0
    return bigram_counts[(prev_word, word)] / unigram_counts[prev_word]

# ----------------------------
# Sentence Probability Function
# ----------------------------
def sentence_probability(sentence):
    words = sentence.split()
    prob = 1.0
    
    for i in range(1, len(words)):
        prev_word = words[i-1]
        word = words[i]
        prob *= bigram_probability(prev_word, word)
    
    return prob

# ----------------------------
# Test Sentences
# ----------------------------
s1 = "<s> I love NLP </s>"
s2 = "<s> I love deep learning </s>"

prob1 = sentence_probability(s1)
prob2 = sentence_probability(s2)

print("Sentence 1 Probability:", prob1)
print("Sentence 2 Probability:", prob2)

if prob1 > prob2:
    print("The model prefers Sentence 1.")
elif prob2 > prob1:
    print("The model prefers Sentence 2.")
else:
    print("Both sentences are equally probable.")
