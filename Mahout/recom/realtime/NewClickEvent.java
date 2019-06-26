package com.dylan.recom.realtime;

/**
 * Created by dylan
 */
public class NewClickEvent {
  private long userId;
  private long itemId;

  public NewClickEvent() {
    this.userId = -1L;
    this.itemId = -1L;
  }

  public NewClickEvent(long userId, long itemId) {
    this.userId = userId;
    this.itemId = itemId;
  }

  public long getUserId() {
    return userId;
  }

  public void setUserId(long userId) {
    this.userId = userId;
  }

  public long getItemId() {
    return itemId;
  }

  public void setItemId(long itemId) {
    this.itemId = itemId;
  }
}
