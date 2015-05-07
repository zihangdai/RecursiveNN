
all: stanford_sentiment

stanfordSentimentTreebank.zip:
	wget http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip

stanfordSentimentTreebankRaw.zip:
	http://nlp.stanford.edu/~socherr/stanfordSentimentTreebankRaw.zip

trainDevTestTrees_PTB.zip:
	wget http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip

stanford_sentiment: trainDevTestTrees_PTB.zip
	mkdir $@
	unzip $<
	mkdir $@/train
	mkdir $@/test
	mv trees/dev.txt $@/dev.mrg
	mv trees/test.txt $@/test/test.mrg
	mv trees/train.txt $@/train/train.mrg
	rm -rf trees

.PHONY: clean

clean:
	rm -rf stanford_sentiment