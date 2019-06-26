package com.dylan.recom.webservice;

import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.servlet.ServletContextHandler;
import org.eclipse.jetty.servlet.ServletHolder;
import org.glassfish.jersey.servlet.ServletContainer;

/**
 * Created by dylan
 */
public class RecoServer {
  private Server webServer = null;
  public void start() {
    ServletContextHandler context = new ServletContextHandler(ServletContextHandler.NO_SESSIONS);
    context.setContextPath("/");

    webServer = new Server(9999);
    webServer.setHandler(context);

    ServletHolder jerseyServlet = context.addServlet(ServletContainer.class, "/*");
    jerseyServlet.setInitOrder(0);

    // Tells the Jersey Servlet which REST service/class to load.
    jerseyServlet.setInitParameter("jersey.config.server.provider.packages",
        "com.dylan.recom.webservice");

    try {
      System.out.println("Web Server started ......");
      webServer.start();
      webServer.join();
    } catch(Exception e) {
      e.printStackTrace();
    } finally {
      webServer.destroy();
    }

  }

  public void stop() throws Exception{
    if(webServer != null) {
      webServer.stop();
    }
  }

  public static void main(String[] args) throws Exception {
    RecoServer recoServer = new RecoServer();
    recoServer.start();
  }
}
