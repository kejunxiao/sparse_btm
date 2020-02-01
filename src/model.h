#ifndef _MODEL_H
#define _MODEL_H

#include "utils.h"

typedef struct TopicNode_ {
    TopicNode_ *prev;
    TopicNode_ *next;
    uint32 cnt;
    uint32 topicid;
} TopicNode;
    
typedef struct DocEntry_ {
    uint32 docid;
    uint32 idx;
    uint32 num_words;
} DocEntry;

typedef struct WordEntry_ {
    uint32 wordid;
    TopicNode *nonzeros;
} WordEntry;

typedef struct TokenEntry_ {
    uint32 wordid;
} TokenEntry;

/* public interface */
void topicNodeInit(TopicNode *topic_node, int topicid);
void docEntryInit(DocEntry *doc_entry, uint32 docid);
void wordEntryInit(WordEntry *word_entry, uint32 wordid);
inline uint32 getTopicWordCnt(TopicNode *topic_word_dist, uint32 num_topics, int topicid, uint32 wordid) {
    return topic_word_dist[wordid * num_topics + topicid].cnt;
}
void addTopicWordCnt(TopicNode *topic_word_dist, uint32 num_topics, int topicid, WordEntry *word_entry, int delta);

#endif //MODEL_H
