# sparse_btm
a cpp implementation of sparse biterm topic model, 10x faster than [origin implementation](https://github.com/xiaohuiyan/BTM) because using sparse-gibbs-sampler.

# features:
* being suitable to model for user-click-sequenece(Rcommandation System) or short-text(NLP), because it assume that adjacent N-items belong to a topic;
* supprot load last-trained-model and continue training;
* using sparse-gibbs-sampler, 10x faster than origin implementation;

# arguments:
_____________________________________

Biterm Topic Model (Sparse-Sampler)  

_____________________________________

Parameters:
* -input <file>  
	path of docs file, lines of file look like "word1 word2 word3 ... \n"  
	word is <string>, freq is <int>, represent word-freqence in the document  
* -output <dir>  
	dir of model(topic_biterm_sum, topic_word) file  
* -num_topics <int>  
	number of topics  
* -alpha <float>  
	symmetric doc-topic prior probability, default is 0.05  
* -beta <float>  
	symmetric topic-word prior probability, default is 0.01  
* -window_size <int>  
	window size for biterms, default is 2  
* -num_iters <int>  
	number of iteration, default is 20  
* -save_step <int>  
	save model every save_step iteration, default is -1 (no save)  
  
# usage:
./sparse_btm -input short_text.txt -output model_out/ -num_topics 100 -window_size 3 -num-iters 20 -save_step 10
