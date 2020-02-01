#include "model.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

void topicNodeInit(TopicNode *topic_node, int topicid) {
    topic_node->prev = NULL;
    topic_node->next = NULL;
    topic_node->cnt = 0;
    topic_node->topicid = topicid;
}

void docEntryInit(DocEntry *doc_entry, uint32 docid) {
    doc_entry->docid = docid;
    doc_entry->idx = 0;
    doc_entry->num_words = 0;
}

void wordEntryInit(WordEntry *word_entry, uint32 wordid) {
    word_entry->wordid = wordid;
    word_entry->nonzeros = NULL;
}

void addTopicWordCnt(TopicNode *topic_word_dist, uint32 num_topics, int topicid, WordEntry *word_entry, int delta) {
    uint32 oldcnt, offset;
    TopicNode *node;

    offset = word_entry->wordid * num_topics + topicid;
    oldcnt = topic_word_dist[offset].cnt;
    topic_word_dist[offset].cnt += delta;

    if (oldcnt == 0 && delta > 0) { 
        // insert topicid into nozeros of wordid
        node = &topic_word_dist[offset];
        node->next = word_entry->nonzeros;
        if (word_entry->nonzeros) (word_entry->nonzeros)->prev = node;
        word_entry->nonzeros = node;
    } else if (topic_word_dist[offset].cnt == 0 && delta < 0) {
        // remove topicid from nonzeros of wordid
        node = &topic_word_dist[offset];
        if (node->prev) node->prev->next = node->next;
        else word_entry->nonzeros = node->next;
        if (node->next) node->next->prev = node->prev;
        node->prev = NULL;
        node->next = NULL;
    }
}
