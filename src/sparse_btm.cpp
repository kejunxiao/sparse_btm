#include <sys/time.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>

#include "utils.h"
#include "model.h"

/* gloabl variables */
// parameters
char input[MAX_STRING];
char output[MAX_STRING];
uint32 num_topics  = 0;
real alpha         = 0.05;
real beta          = 0.01;
uint32 window_size = 2;
uint32 num_iters   = 20;
int save_step      = -1;
char init_tb_model[MAX_STRING];
char init_tw_model[MAX_STRING];

// train data related
uint32 num_docs   = 0;
uint32 vocab_size = 0;
uint32 num_tokens = 0;
std::unordered_map<std::string, uint32> word2id;
std::unordered_map<uint32, std::string> id2word;

// model related
uint32 *topic_word_sums    = NULL;
uint32 *topic_biterm_sums  = NULL;
TopicNode *topic_word_dist = NULL;

DocEntry *doc_entries      = NULL;
WordEntry *word_entries    = NULL;
TokenEntry *token_entries  = NULL;

std::vector<int> biterm_topics;

// init model related
uint32 *init_topic_word_sums    = NULL;
uint32 *init_topic_biterm_sums  = NULL;
TopicNode *init_topic_word_dist = NULL;

WordEntry *init_word_entries    = NULL;

std::unordered_set<uint32> init_word_set;

/* helper functions */
static void getWordFromId(uint32 wordid, char *word) {
    std::unordered_map<uint32, std::string>::iterator itr = id2word.find(wordid);
    if (itr != id2word.end()) {
        strcpy(word, itr->second.c_str());
        return;
    } else { 
        fprintf(stderr, "***ERROR***: unknown wordid %d", wordid); 
        exit(1);
    }
}

static uint32 getIdFromWord(const char *word) {
    std::string s(word);
    std::unordered_map<std::string, uint32>::iterator itr = word2id.find(s);
    if (itr != word2id.end()) {
        return itr->second;
    } else {
        word2id[s] = vocab_size;
        id2word[vocab_size] = s;
        vocab_size++;
        return vocab_size;
    }
}

inline static int genRandTopicId() { return rand() % num_topics; }

/* sparse LDA process */
// denominators
static void initDenomin(real *denominators, real Vbeta) {
    int t;
    for (t = 0; t < num_topics; t++) denominators[t] = (topic_word_sums[t] + Vbeta) * (topic_word_sums[t] + Vbeta) / (topic_biterm_sums[t] + alpha);
}

inline static void updateDenomin(real *denominators, real Vbeta, int topicid) {
    denominators[topicid] = (topic_word_sums[topicid] + Vbeta) * (topic_word_sums[topicid] + Vbeta) / (topic_biterm_sums[topicid] + alpha);
}

static void initDenominByInitModel(real *denominators, real Vbeta) {
    int t;
    for (t = 0; t < num_topics; t++) denominators[t] = (init_topic_word_sums[t] + Vbeta) * (init_topic_word_sums[t] + Vbeta) / (init_topic_biterm_sums[t] + alpha);
}

// soomth-only bucket
static real initS(real *sbucket, real bb, real *denominators) {
    int t;
    real smooth = 0;

    for (t = 0; t < num_topics; t++) {
        sbucket[t] = bb / denominators[t];
        smooth += sbucket[t];
    }
    return smooth;
}

static real updateS(real *sbucket, real bb, real *denominators, int topicid) {
    real delta = 0, tmp = 0;

    tmp = bb / denominators[topicid];
    delta += tmp - sbucket[topicid];
    sbucket[topicid] = tmp;
    return delta;
}

// word1 bucket, n(w1|t) * (n(w2|t) + beta)
static real initW1(real *w1bucket, WordEntry *word_entry1, WordEntry *word_entry2, real *denominators) {
    TopicNode *node;
    real w1sum = 0;

    node = word_entry1->nonzeros;
    while (node) {
        w1bucket[node->topicid] = node->cnt *(getTopicWordCnt(topic_word_dist, num_topics, node->topicid, word_entry2->wordid) + beta) / denominators[node->topicid];
        w1sum += w1bucket[node->topicid];
        node = node->next;
    }
    return w1sum;
}

static real initW1ByInitModel(real *w1bucket, WordEntry *word_entry1, WordEntry *word_entry2, real *denominators) {
    TopicNode *node;
    real w1sum = 0;

    node = word_entry1->nonzeros;
    while (node) {
        w1bucket[node->topicid] = node->cnt *(getTopicWordCnt(init_topic_word_dist, num_topics, node->topicid, word_entry2->wordid) + beta) / denominators[node->topicid];
        w1sum += w1bucket[node->topicid];
        node = node->next;
    }
    return w1sum;
}

// word2 bucket, n(w2|t) * beta
static real initW2(real *w2bucket, WordEntry *word_entry2, real *denominators) {
    TopicNode *node;
    real w2sum = 0;

    node = word_entry2->nonzeros;
    while (node) {
        w2bucket[node->topicid] = node->cnt * beta / denominators[node->topicid];
        w2sum += w2bucket[node->topicid];
        node = node->next;
    }
    return w2sum;
}

/* public interface */
void learnVocabFromDocs() {
    uint32 a, len;
    char ch, buf[MAX_STRING];
    FILE *fin;

    if (NULL == (fin = fopen(input, "r"))) {
        fprintf(stderr, "***ERROR***: can not open %s file", input);
        exit(1);
    }
    // get number of documents and number of tokens from input file
    len = 0;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == ' ' || ch == '\n') {
            buf[len] = '\0';
            getIdFromWord(buf);
            num_tokens++;
            memset(buf, 0, len);
            len = 0;
            if (ch == '\n') {
                num_docs++;
                if (num_docs % 1000 == 0) {
                    printf("%dK%c", num_docs / 1000, 13);
                    fflush(stdout);
                }
            }
        } else { // append ch to buf
            if (len == MAX_STRING - 1) continue;
            buf[len] = ch;
            len++;
        }
    }
    printf("number of documents: %d, number of tokens: %d, vocabulary size: %d\n", num_docs, num_tokens, vocab_size);

    // allocate memory for topic-word distribution
    topic_word_dist = (TopicNode *)calloc(vocab_size * num_topics, sizeof(TopicNode));
    for (a = 0; a < vocab_size * num_topics; a++) topicNodeInit(&topic_word_dist[a], a % num_topics);
    // allocate memory for doc_entries
    doc_entries = (DocEntry *)calloc(num_docs, sizeof(DocEntry));
    for (a = 0; a < num_docs; a++) docEntryInit(&doc_entries[a], a);
    // allocate memory for word_entries
    word_entries = (WordEntry *)calloc(vocab_size, sizeof(WordEntry));
    for (a = 0; a < vocab_size; a++) wordEntryInit(&word_entries[a], a);
    // allocate memory for token_entries
    token_entries = (TokenEntry *)calloc(num_tokens, sizeof(TokenEntry));
    // reallocate storage for biterm_topics
    biterm_topics.reserve(num_tokens);
}

void loadDocs() {
    uint32 a, b, len, wordid, docid;
    char ch, buf[MAX_STRING];
    FILE *fin;
    DocEntry *doc_entry;
    TokenEntry *token_entry;

    if (NULL == (fin = fopen(input, "r"))) {
        fprintf(stderr, "***ERROR***: can not open %s file", input);
        exit(1);
    }
    // load documents
    docid = 0;
    len = 0;
    a = 0;
    b = 0;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == ' ' || ch == '\n') {
            buf[len] = '\0';
            wordid = getIdFromWord(buf);

            doc_entry = &doc_entries[docid];
            token_entry = &token_entries[a];
            token_entry->wordid = wordid;
            a++;
            memset(buf, 0, len);
            len = 0;
            if (ch == '\n') {
                doc_entry = &doc_entries[docid];
                doc_entry->idx = b;
                doc_entry->num_words = a - b;

                docid++;
                b = a;
                if (docid % 1000 == 0) {
                    printf("%dK%c", docid / 1000, 13);
                    fflush(stdout);
                }
            }
        } else { // append ch to buf
            if (len == MAX_STRING - 1) continue;
            buf[len] = ch;
            len++;
        }
    }
}

void allocateInitModel() {
    uint32 a;

    // allocate memory for init-topic-word-sums
    init_topic_word_sums = (uint32 *)calloc(num_topics, sizeof(uint32));
    memset(init_topic_word_sums, 0, num_topics * sizeof(uint32));
    // allocate memory for init-topic-biterm-sums
    init_topic_biterm_sums = (uint32 *)calloc(num_topics, sizeof(uint32));
    memset(init_topic_biterm_sums, 0, num_topics * sizeof(uint32));
    // allocate memory for init-topic-word distribution
    init_topic_word_dist = (TopicNode *)calloc(vocab_size * num_topics, sizeof(TopicNode));
    for (a = 0; a < vocab_size * num_topics; a++) topicNodeInit(&init_topic_word_dist[a], a % num_topics);
    // allocate memory for init_word_entries
    init_word_entries = (WordEntry *)calloc(vocab_size, sizeof(WordEntry));
    for (a = 0; a < vocab_size; a++) wordEntryInit(&init_word_entries[a], a);
}

void loadInitModel() {
    int topicid;
    uint32 is_newline, len, cnt, wordid;
    char ch, *token, buf[MAX_STRING];
    WordEntry *word_entry;
    FILE *fin;

    // load topic-biterm-sums
    if (NULL == (fin = fopen(init_tb_model, "r"))) {
        fprintf(stderr, "***ERROR***: can not open %s file", init_tb_model);
        exit(1);
    }
    len = 0;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == ' ' || ch == '\n') {
            buf[len] = '\0';
            token = strtok(buf, ":");  // get topicid
            topicid = atoi(token);
            token = strtok(NULL, ":"); // get biterm-cnt-sum
            cnt = atoi(token);
            init_topic_biterm_sums[topicid] = cnt;
            memset(buf, 0, len);
            len = 0;
        } else { // append ch to buf
            if (len == MAX_STRING - 1) continue;
            buf[len] = ch;
            len++;
        }
    }
    // load topic-word
    if (NULL == (fin = fopen(init_tw_model, "r"))) {
        fprintf(stderr, "***ERROR***: can not open %s file", init_tw_model);
        exit(1);
    }
    len = 0;
    is_newline = 1;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == ' ' || ch == '\n') {
            buf[len] = '\0';
            if (is_newline) {
                topicid = atoi(buf);
                is_newline = 0;
            } else {
                token = strtok(buf, ":");  // get word
                std::string word(token);
                auto itr = word2id.find(word);
                if (itr != word2id.end()) {
                    wordid = getIdFromWord(token);
                    word_entry = &init_word_entries[wordid];
                    token = strtok(NULL, ":"); // get topic-word-cnt
                    cnt = atoi(token);
                    addTopicWordCnt(init_topic_word_dist, num_topics, topicid, word_entry, cnt);
                    init_topic_word_sums[topicid] += cnt;
                    init_word_set.insert(wordid);
                }
            }
            memset(buf, 0, len);
            len = 0;
            if (ch == '\n') {
                is_newline = 1;
            }
        } else { // append ch to buf
            if (len == MAX_STRING - 1) continue;
            buf[len] = ch;
            len++;
        }
    }
}

void initBitermsByInitModel() {
    uint32 a, b, c;
    int topicid;
    real smooth, w1sum, w2sum, r, *denominators, *sbucket, *w1bucket, *w2bucket;
    real Vbeta = vocab_size * beta, bb = beta * beta;
    struct timeval tv1, tv2;
    std::string word1, word2;
    DocEntry *doc_entry;
    TokenEntry *token_entry1, *token_entry2;
    WordEntry *word_entry1, *word_entry2, *init_word_entry1, *init_word_entry2;
    TopicNode *node;

    denominators = (real *)calloc(num_topics, sizeof(real));
    sbucket = (real *)calloc(num_topics, sizeof(real));
    w1bucket = (real *)calloc(num_topics, sizeof(real));
    w2bucket = (real *)calloc(num_topics, sizeof(real));

    memset(denominators, 0, num_topics * sizeof(real));
    initDenominByInitModel(denominators, Vbeta);
    memset(sbucket, 0, num_topics * sizeof(real));
    smooth = initS(sbucket, bb, denominators);

    // iterate documents
    printf("start init biterms by init model:");
    gettimeofday(&tv1, NULL);
    for (a = 0; a < num_docs; a++) { // a is docid
        if (a > 0 && a % 10000 == 0) {
            gettimeofday(&tv2, NULL);
            printf("%cProcess: %.2f%% Documents/Sec: %.2fK",
                   13,
                   (a * 1. / num_docs) * 100.,
                   10. / (tv2.tv_sec - tv1.tv_sec + (tv2.tv_usec - tv1.tv_usec) / 1000000.));
            fflush(stdout);
            memcpy(&tv1, &tv2, sizeof(struct timeval));
        }
        doc_entry = &doc_entries[a];
        // iterate biterms 
        for (b = 0; b < doc_entry->num_words - 1; b++) {
            for (c = b + 1; c < b + window_size && c < doc_entry->num_words; c++) {
                token_entry1 = &token_entries[doc_entry->idx + b];
                word_entry1 = &word_entries[token_entry1->wordid];
                init_word_entry1 = &init_word_entries[token_entry1->wordid];
                token_entry2 = &token_entries[doc_entry->idx + c];
                word_entry2 = &word_entries[token_entry2->wordid];
                init_word_entry2 = &init_word_entries[token_entry2->wordid];
                // assign topic to biterm by init model
                auto itr1 = init_word_set.find(word_entry1->wordid);
                auto itr2 = init_word_set.find(word_entry2->wordid);
                if (itr1 == init_word_set.end() and itr2 == init_word_set.end()) {
                    topicid = genRandTopicId();
                } else if (itr1 != init_word_set.end() and itr2 != init_word_set.end()) {
                    // calc word1-bucket and word2-bucket
                    memset(w1bucket, 0, num_topics * sizeof(real));
                    w1sum = initW1ByInitModel(w1bucket, init_word_entry1, init_word_entry2, denominators);
                    memset(w2bucket, 0, num_topics * sizeof(real));
                    w2sum = initW2(w2bucket, init_word_entry2, denominators);
                    // start sampling
                    topicid = -1;
                    r = (smooth + w1sum + w2sum) * rand() / (RAND_MAX + 1.);
                    if (r < smooth) {
                        for (topicid = 0; topicid < num_topics; topicid++) {
                            if (r < sbucket[topicid]) break;
                            r -= sbucket[topicid];
                        }
                    } else if (r < smooth + w1sum) {
                        r -= smooth;
                        node = init_word_entry1->nonzeros;
                        while (node) {
                            if (r < w1bucket[node->topicid]) {topicid = node->topicid; break;}
                            r -= w1bucket[node->topicid];
                            node = node->next;
                        }
                    } else {
                        r -= smooth + w1sum;
                        node = init_word_entry2->nonzeros;
                        while (node) {
                            if (r < w2bucket[node->topicid]) {topicid = node->topicid; break;}
                            r -= w2bucket[node->topicid];
                            node = node->next;
                        }
                    }
                } else if (itr1 != init_word_set.end()) {
                    memset(w2bucket, 0, num_topics * sizeof(real));
                    w2sum = initW2(w2bucket, init_word_entry1, denominators);
                    // start sampling
                    topicid = -1;
                    r = (smooth + w2sum) * rand() / (RAND_MAX + 1.);
                    if (r < smooth) {
                        for (topicid = 0; topicid < num_topics; topicid++) {
                            if (r < sbucket[topicid]) break;
                            r -= sbucket[topicid];
                        }
                    } else {
                        r -= smooth;
                        node = init_word_entry1->nonzeros;
                        while (node) {
                            if (r < w2bucket[node->topicid]) {topicid = node->topicid; break;}
                            r -= w2bucket[node->topicid];
                            node = node->next;
                        }
                    }
                } else {
                    memset(w2bucket, 0, num_topics * sizeof(real));
                    w2sum = initW2(w2bucket, init_word_entry2, denominators);
                    // start sampling
                    topicid = -1;
                    r = (smooth + w2sum) * rand() / (RAND_MAX + 1.);
                    if (r < smooth) {
                        for (topicid = 0; topicid < num_topics; topicid++) {
                            if (r < sbucket[topicid]) break;
                            r -= sbucket[topicid];
                        }
                    } else {
                        r -= smooth;
                        node = init_word_entry2->nonzeros;
                        while (node) {
                            if (r < w2bucket[node->topicid]) {topicid = node->topicid; break;}
                            r -= w2bucket[node->topicid];
                            node = node->next;
                        }
                    }
                }
                if (topicid < 0) {
                    fprintf(stderr, "***ERROR***: sample fail");
                    fflush(stderr);
                    exit(2);
                }
                // assign new topicid
                biterm_topics.push_back(topicid);
                addTopicWordCnt(topic_word_dist, num_topics, topicid, word_entry1, 1);
                addTopicWordCnt(topic_word_dist, num_topics, topicid, word_entry2, 1);
                topic_biterm_sums[topicid]++;
                topic_word_sums[topicid] += 2;
            }
        }
    }
    free(denominators);
    free(sbucket);
    free(w1bucket);
    free(w2bucket);
    printf("\nfinished init biterms.\n");
}

void freeInitModel() {
    free(init_topic_word_sums); 
    free(init_topic_biterm_sums);
    free(init_topic_word_dist);
    free(init_word_entries); 
}

void initBiterms() {
    uint32 a, b, c;
    int topicid;
    struct timeval tv1, tv2;
    DocEntry *doc_entry;
    TokenEntry *token_entry1, *token_entry2;
    WordEntry *word_entry1, *word_entry2;

    printf("start init biterms:");
    gettimeofday(&tv1, NULL);
    for (a = 0; a < num_docs; a++) {
        if (a > 0 && a % 10000 == 0) {
            gettimeofday(&tv2, NULL);
            printf("%cProcess: %.2f%% Documents/Sec: %.2fK",
                   13,
                   (a * 1. / num_docs) * 100.,
                   10. / (tv2.tv_sec - tv1.tv_sec + (tv2.tv_usec - tv1.tv_usec) / 1000000.));
            fflush(stdout);
            memcpy(&tv1, &tv2, sizeof(struct timeval));
        }
        doc_entry = &doc_entries[a];
        for (b = 0; b < doc_entry->num_words - 1; b++) {
            for (c = b + 1; c < b + window_size && c < doc_entry->num_words; c++) {
                token_entry1 = &token_entries[doc_entry->idx + b];
                word_entry1 = &word_entries[token_entry1->wordid];
                token_entry2 = &token_entries[doc_entry->idx + c];
                word_entry2 = &word_entries[token_entry2->wordid];
                // assign topic to biterm randomly
                topicid = genRandTopicId();
                biterm_topics.push_back(topicid);
                addTopicWordCnt(topic_word_dist, num_topics, topicid, word_entry1, 1);
                addTopicWordCnt(topic_word_dist, num_topics, topicid, word_entry2, 1);
                topic_biterm_sums[topicid]++;
                topic_word_sums[topicid] += 2;
            }
        }
    }
    printf("\nfinished init biterms.\n");
}

void gibbsSample(uint32 round) {
    uint32 a, b, c, bidx;
    int old_topicid, new_topicid;
    real smooth, w1sum, w2sum, r, *denominators, *sbucket, *w1bucket, *w2bucket;
    real Vbeta = vocab_size * beta, bb = beta * beta;
    struct timeval tv1, tv2;
    DocEntry *doc_entry;
    TokenEntry *token_entry1, *token_entry2;
    WordEntry *word_entry1, *word_entry2;
    TopicNode *node;

    denominators = (real *)calloc(num_topics, sizeof(real));
    sbucket = (real *)calloc(num_topics, sizeof(real));
    w1bucket = (real *)calloc(num_topics, sizeof(real));
    w2bucket = (real *)calloc(num_topics, sizeof(real));

    memset(denominators, 0, num_topics * sizeof(real));
    initDenomin(denominators, Vbeta);
    memset(sbucket, 0, num_topics * sizeof(real));
    smooth = initS(sbucket, bb, denominators);
    gettimeofday(&tv1, NULL);
    bidx = 0;
    // iterate documents
    for (a = 0; a < num_docs; a++) { // a is docid
        if (a > 0 && a % 10000 == 0) {
            gettimeofday(&tv2, NULL);
            printf("%cProcess: %.2f%% Documents/Sec: %.2fK",
                   13,
                   (round + a * 1. / num_docs) * 100. / num_iters,
                   10. / (tv2.tv_sec - tv1.tv_sec + (tv2.tv_usec - tv1.tv_usec) / 1000000.));
            fflush(stdout);
            memcpy(&tv1, &tv2, sizeof(struct timeval));
        }
        doc_entry = &doc_entries[a];
        // iterate biterms 
        for (b = 0; b < doc_entry->num_words - 1; b++) {
            for (c = b + 1; c < b + window_size && c < doc_entry->num_words; c++) {
                token_entry1 = &token_entries[doc_entry->idx + b];
                word_entry1 = &word_entries[token_entry1->wordid];
                token_entry2 = &token_entries[doc_entry->idx + c];
                word_entry2 = &word_entries[token_entry2->wordid];
                old_topicid = biterm_topics[bidx];

                // remove old topicid
                addTopicWordCnt(topic_word_dist, num_topics, old_topicid, word_entry1, -1);
                addTopicWordCnt(topic_word_dist, num_topics, old_topicid, word_entry2, -1);
                topic_word_sums[old_topicid] -= 2;
                topic_biterm_sums[old_topicid]--;

                // update denominator and smooth-only bucket
                updateDenomin(denominators, Vbeta, old_topicid);
                smooth += updateS(sbucket, bb, denominators, old_topicid);

                // calc word1-bucket and word2-bucket
                memset(w1bucket, 0, num_topics * sizeof(real));
                w1sum = initW1(w1bucket, word_entry1, word_entry2, denominators);
                memset(w2bucket, 0, num_topics * sizeof(real));
                w2sum = initW2(w2bucket, word_entry2, denominators);

                // start sampling
                new_topicid = -1;
                r = (smooth + w1sum + w2sum) * rand() / (RAND_MAX + 1.);
                if (r < smooth) {
                    for (new_topicid = 0; new_topicid < num_topics; new_topicid++) {
                        if (r < sbucket[new_topicid]) break;
                        r -= sbucket[new_topicid];
                    }
                } else if (r < smooth + w1sum) {
                    r -= smooth;
                    node = word_entry1->nonzeros;
                    while (node) {
                        if (r < w1bucket[node->topicid]) {new_topicid = node->topicid; break;}
                        r -= w1bucket[node->topicid];
                        node = node->next;
                    }
                } else {
                    r -= smooth + w1sum;
                    node = word_entry2->nonzeros;
                    while (node) {
                        if (r < w2bucket[node->topicid]) {new_topicid = node->topicid; break;}
                        r -= w2bucket[node->topicid];
                        node = node->next;
                    }
                }
                if (new_topicid < 0) {
                    fprintf(stderr, "***ERROR***: sample fail, r = %.16f, smooth = %.16f, w1sum = %.16f, w2sum = %.16f\n", r, smooth, w1sum, w2sum);
                    fflush(stderr);
                    exit(2);
                }
                // assign new topicid
                addTopicWordCnt(topic_word_dist, num_topics, new_topicid, word_entry1, 1);
                addTopicWordCnt(topic_word_dist, num_topics, new_topicid, word_entry2, 1);
                topic_word_sums[new_topicid] += 2;
                biterm_topics[bidx++] = new_topicid;
                topic_biterm_sums[new_topicid]++;
                // update denominator and smooth-only bucket
                updateDenomin(denominators, Vbeta, new_topicid);
                smooth += updateS(sbucket, bb, denominators, new_topicid);
            }
        }
    }
    free(denominators);
    free(sbucket);
    free(w1bucket);
    free(w2bucket);
}

void saveModel(uint32 suffix) {
    uint32 a, cnt;
    int t;
    char fpath[MAX_STRING], word_str[MAX_STRING];
    FILE *fout;

    // save topic-biterm-sums
    sprintf(fpath, "%s/%s.%d", output, "topic_biterm_sum", suffix);
    if (NULL == (fout = fopen(fpath, "w"))) {
        fprintf(stderr, "***ERROR***: open %s fail", fpath);
        exit(1);
    }
    for (t = 0; t < num_topics; t++) {
        if (t == 0) {
            fprintf(fout, "%d:%d", t, topic_biterm_sums[t]);
        } else {
            fprintf(fout, " %d:%d", t, topic_biterm_sums[t]);
        }
    }
    fprintf(fout, "\n");
    fflush(fout);

    // save topic-word
    sprintf(fpath, "%s/%s.%d", output, "topic_word", suffix);
    if (NULL == (fout = fopen(fpath, "w"))) {
        fprintf(stderr, "***ERROR***: open %s fail", fpath);
        exit(1);
    }
    for (t = 0; t < num_topics; t++) {
        fprintf(fout, "%d", t);
        for (a = 0; a < vocab_size; a++) {
            cnt = getTopicWordCnt(topic_word_dist, num_topics, t, a);
            if (cnt > 0) {
                getWordFromId(a, word_str);
                fprintf(fout, " %s:%d", word_str, cnt);
                memset(word_str, 0, MAX_STRING);
            }
        }
        fprintf(fout, "\n");
    }
    fflush(fout);
}

int main(int argc, char **argv) {
    int a;

    if (argc == 1) {
        printf("_____________________________________\n\n");
        printf("Biterm Topic Model (Sparse-Sampler)  \n\n");
        printf("_____________________________________\n\n");
        printf("Parameters:\n");
        printf("-input <file>\n");
        printf("\tpath of docs file, lines of file look like \"word1 word2 word3 ... \\n\", word is <string>.\n");
        printf("-output <dir>\n");
        printf("\tdir of model(topic_biterm_sum, topic_word) file\n");
        printf("-num_topics <int>\n");
        printf("\tnumber of topics\n");
        printf("-alpha <float>\n");
        printf("\tsymmetric doc-topic prior probability, default is 0.05\n");
        printf("-beta <float>\n");
        printf("\tsymmetric topic-word prior probability, default is 0.01\n");
        printf("-window_size <int>\n");
        printf("\twindow size for biterms, default is 2\n");
        printf("-num_iters <int>\n");
        printf("\tnumber of iteration, default is 20\n");
        printf("-save_step <int>\n");
        printf("\tsave model every save_step iteration, default is -1 (no save)\n");
        printf("-init_tb <file>\n");
        printf("\tpath of init topic-biterm-sums\n");
        printf("-init_tw <file>\n");
        printf("\tpath of init topic-word\n");
        return -1;
    }

    // parse args
    if ((a = argPos((char *)"-input", argc, argv)) > 0) {
        strcpy(input, argv[a + 1]);
    }
    if ((a = argPos((char *)"-output", argc, argv)) > 0) {
        strcpy(output, argv[a + 1]);
    }
    if ((a = argPos((char *)"-num_topics", argc, argv)) > 0) {
        num_topics = atoi(argv[a + 1]);
    }
    if ((a = argPos((char *)"-alpha", argc, argv)) > 0) {
        alpha = atof(argv[a + 1]);
    }
    if ((a = argPos((char *)"-beta", argc, argv)) > 0) {
        beta = atof(argv[a + 1]);
    }
    if ((a = argPos((char *)"-window_size", argc, argv)) > 0) {
        window_size = atoi(argv[a + 1]);
    }
    if ((a = argPos((char *)"-num_iters", argc, argv)) > 0) {
        num_iters = atoi(argv[a + 1]);
    }
    if ((a = argPos((char *)"-save_step", argc, argv)) > 0) {
        save_step = atoi(argv[a + 1]);
    }
    if ((a = argPos((char *)"-init_tb", argc, argv)) > 0) {
        strcpy(init_tb_model, argv[a + 1]);
    }
    if ((a = argPos((char *)"-init_tw", argc, argv)) > 0) {
        strcpy(init_tw_model, argv[a + 1]);
    }

    // allocate memory for topic-word-sums
    topic_word_sums = (uint32 *)calloc(num_topics, sizeof(uint32));
    memset(topic_word_sums, 0, num_topics * sizeof(uint32));
    // allocate memory for topic-biterm-sums
    topic_biterm_sums = (uint32 *)calloc(num_topics, sizeof(uint32));
    memset(topic_biterm_sums, 0, num_topics * sizeof(uint32));

    srand(time(NULL));

    // load documents and allocate memory for entries
    learnVocabFromDocs();
    loadDocs();
    // init biterms
    if (strlen(init_tb_model) > 0 && strlen(init_tw_model) > 0) {
        allocateInitModel();
        loadInitModel();
        initBitermsByInitModel();
        freeInitModel();
    } else {
        initBiterms();
    }

    // gibbs sampling
    printf("start train LDA:\n");
    for (a = 0; a < num_iters; a++) {
        if (save_step > 0 && a % save_step == 0) saveModel(a);
        gibbsSample(a);
    }

    // save model
    saveModel(num_iters);

    free(topic_word_sums);
    free(topic_biterm_sums);
    free(topic_word_dist);
    free(doc_entries);
    free(word_entries);
    free(token_entries);

    return 0;
}
