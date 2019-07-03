package com.dylan.recom.common;

/**
 * Created by dylan
 */
public class ItemSimilarity implements Comparable<ItemSimilarity> {
  private long id; //itemID
  private Double s; //similarity

  public ItemSimilarity() {
    this.id = -1;
    this.s = 0d;
  }
  public ItemSimilarity(long itemId, Double similarity) {
    this.id = itemId;
    this.s = similarity;
  }

  public long getId() {
    return id;
  }

  public void setId(long itemId) {
    this.id = itemId;
  }

  public Double getS(){
    return s;
  }

  public void setS(Double similarity) {
    this.s = similarity;
  }

  public boolean equals(Object obj) {
    if (!(obj instanceof ItemSimilarity))
      return false;
    if (obj == this)
      return true;

    // TODO: double number should not compare directly
    return this.id == ((ItemSimilarity) obj).id && this.s == ((ItemSimilarity) obj).s;
  }

  public int hashCode(){
    return (int)(id + s);
  }

  @Override
  public int compareTo(ItemSimilarity obj) {
    if(this.s > obj.s) {
      return 1;
    } else if(this.s < obj.s) {
      return -1;
    }
    return 0;
  }

  @Override
  public String toString() {
    return "id:" + id + ",similarity:" + s;
  }
}
