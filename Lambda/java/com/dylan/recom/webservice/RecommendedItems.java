package com.dylan.recom.webservice;

/**
 * Created by dylan
 */
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;

@XmlRootElement
public class RecommendedItems {
  private Long[] items = null;
  public Long[] getItems() {
    return items;
  }
  public void setItems(Long[] items) {
    this.items = items;
  }
}
