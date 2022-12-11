package org.batfish.backend;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.SortedMap;
import java.util.SortedSet;
import java.util.TreeSet;
import javax.annotation.Nullable;
import org.batfish.common.BatfishException;
import org.batfish.common.BdpOscillationException;
import org.batfish.common.bdd.BDDPacket;
import org.batfish.common.plugin.DataPlanePlugin.ComputeDataPlaneResult;
import org.batfish.config.Settings;
import org.batfish.datamodel.AbstractRoute;
import org.batfish.datamodel.AclIpSpace;
import org.batfish.datamodel.Configuration;
import org.batfish.datamodel.DataPlane;
import org.batfish.datamodel.Edge;
import org.batfish.datamodel.EmptyIpSpace;
import org.batfish.datamodel.Fib;
import org.batfish.datamodel.HeaderSpace;
import org.batfish.datamodel.Interface;
import org.batfish.datamodel.InterfaceAddress;
import org.batfish.datamodel.Ip;
import org.batfish.datamodel.IpAccessList;
import org.batfish.datamodel.IpAccessListLine;
import org.batfish.datamodel.IpWildcard;
import org.batfish.datamodel.LocalRoute;
import org.batfish.datamodel.Prefix;
import org.batfish.datamodel.Topology;
import org.batfish.datamodel.collections.NodeInterfacePair;
import org.batfish.datamodel.questions.smt.HeaderLocationQuestion;
import org.batfish.main.Batfish;
import org.batfish.question.SmtNumPathsQuestionPlugin.NumPathsQuestion;
import org.batfish.question.SmtReachabilityQuestionPlugin.ReachabilityQuestion;
import org.batfish.question.SmtWaypointQuestionPlugin.WaypointsQuestion;
import org.batfish.symbolic.GraphEdge;
import org.batfish.symbolic.smt.PropertyChecker;
import org.batfish.symbolic.utils.PathRegexes;
import org.batfish.symbolic.utils.PatternUtils;
import org.batfish.symbolic.utils.Tuple;
import org.jgrapht.Graph;
import org.jgrapht.GraphPath;
import org.jgrapht.alg.shortestpath.AllDirectedPaths;
import org.jgrapht.graph.DefaultDirectedGraph;
import org.jgrapht.graph.DefaultEdge;
import spark.Request;
import spark.Response;

class myNode {
  ArrayList<String> brokenLinks;
  String bestLinks;
  int depth;

  myNode(ArrayList<String> brokenLinks, String bestLinks, int depth){
    this.brokenLinks = brokenLinks;
    this.bestLinks = bestLinks;
    this.depth = depth;
  }

}

public class Server {
  private Batfish batfish;
  private Settings settings;
  private Path scenarioPath;
  private SortedMap<String, Configuration> configurations;
  private int numDataplanes;
  private org.batfish.symbolic.Graph graph;

  private ArrayList<Tuple<String, String>> brokenHotEdgeList = new ArrayList<>();




  private void IterateHotEdges(myNode root, Topology topology, Prefix prefix, String source,
      String destination, ArrayList<Tuple<String, String>> brokenHotEdgeList){

    Deque<myNode> stack = new ArrayDeque<>();
    stack.push(root);
    Map<String, SortedSet<Edge>> nodeEdges = topology.getNodeEdges();
    while(!stack.isEmpty()){
      root = stack.pop();
      if (root.depth==0){
        brokenHotEdgeList.add(new Tuple<>(String.join(",", root.brokenLinks) , root.bestLinks));
        continue;
      }
      for(String link: root.bestLinks.split(",")){
        Set<Edge> hotEdges = new HashSet<>();
        ArrayList<String> totalBrokenLinks = root.brokenLinks;
        totalBrokenLinks.add(link);
        Topology tmpTopology = new Topology(topology.getEdges());

        for(String brokenEdge: totalBrokenLinks){
          String [] node = brokenEdge.split("=");
          Set<Edge> tmpEdges1 = nodeEdges.get(node[0]);
          Set<Edge> tmpEdges2 = nodeEdges.get(node[1]);

          TreeSet<Edge> intersection = new TreeSet<Edge>(tmpEdges1);
          intersection.retainAll(tmpEdges2);
          hotEdges.addAll(intersection);
        }
        tmpTopology.prune(hotEdges, null, null);
        String newBestLinks = getSingleHotEdges(tmpTopology, prefix, source, destination);

        ArrayList<String> childBrokenLinks = root.brokenLinks;
        childBrokenLinks.add(link);

        myNode child = new myNode(childBrokenLinks, newBestLinks, root.depth-1);
        stack.push(child);
      }
    }
  }

  private void Test(String bestLinks, Topology topology, int depth,
      ArrayList<String> brokenEdgeList, Prefix prefix, String source, String destination){

    if (depth==0){
      brokenHotEdgeList.add(new Tuple<String, String>(String.join(",", brokenEdgeList) , bestLinks));
      return;
    }

    Map<String, SortedSet<Edge>> nodeEdges = topology.getNodeEdges();
    String [] edges = bestLinks.split(",");

    Set<Edge> hotEdges = new HashSet<>();
    for (String edge: edges){

      Topology tmpTopology = new Topology(topology.getEdges());

      String [] node = edge.split("=");
      Set<Edge> tmpEdges1 = nodeEdges.get(node[0]);
      Set<Edge> tmpEdges2 = nodeEdges.get(node[1]);

      TreeSet<Edge> intersection = new TreeSet<Edge>(tmpEdges1);
      intersection.retainAll(tmpEdges2);
      hotEdges.addAll(intersection);

      tmpTopology.prune(hotEdges, null, null);

      String newBestLinks = getSingleHotEdges(tmpTopology, prefix, source, destination);
      brokenEdgeList.add(node[0] + "=" + node[1]);

      Test(newBestLinks, tmpTopology, depth-1, brokenEdgeList, prefix, source, destination);

      hotEdges.clear();
      brokenEdgeList.remove(brokenEdgeList.size()-1);
    }

  }


  private String initMinesweeper(String req) {

    String answer;
    Path basePath = null;
    Path containerPath = null;
    String path = "";
    String[] configs = null;

    List<String> initParameters = Arrays.asList(req.split(";"));

    for (String parameter : initParameters) {
      String[] item = parameter.split(":");
      if (item.length > 1) {
        if (item[0].equals("BasePath")) {
          basePath = Paths.get(item[1]);
          containerPath = basePath.resolve("containers/");
        } else if (item[0].equals("ConfigPath")) {
          path = item[1];
        } else if (item[0].equals("ConfigFiles")) {
          configs = item[1].split(",");
        }  else {
          System.out.println("Unknown Parameter: " + item[0] + " - " + item[1]);
        }
      }
    }

    if (basePath != null && containerPath != null && path != "" && configs != null) {
      String[] fullConfigs = new String[configs.length];
      for (int i = 0; i < configs.length; i++) {
        fullConfigs[i] = path + "/" + configs[i];
      }

      batfish = BatfishHelper.getBatfishFromTestrigText(containerPath, fullConfigs);
      settings = batfish.getSettings();
      scenarioPath = Paths.get(path).getParent();

      configurations = batfish.loadConfigurations();

      numDataplanes = 0;

      graph = new org.batfish.symbolic.Graph(batfish);

      answer = "Success";
    } else {
      answer = "Failure";
    }

    return answer;
  }


  public String initMinesweeper(Request req, Response res) {

    String answer;
    Path basePath = null;
    Path containerPath = null;
    String path = "";
    String[] configs = null;

    List<String> initParameters = Arrays.asList(req.body().split(";"));

    for (String parameter : initParameters) {
      String[] item = parameter.split(":");
      if (item.length > 1) {
        if (item[0].equals("BasePath")) {
          basePath = Paths.get(item[1]);
          containerPath = basePath.resolve("containers/");
        } else if (item[0].equals("ConfigPath")) {
          path = item[1];
        } else if (item[0].equals("ConfigFiles")) {
          configs = item[1].split(",");
        }  else {
          System.out.println("Unknown Parameter: " + item[0] + " - " + item[1]);
        }
      }
    }

    if (basePath != null && containerPath != null && path != "" && configs != null) {
      String[] fullConfigs = new String[configs.length];
      for (int i = 0; i < configs.length; i++) {
        fullConfigs[i] = path + "/" + configs[i];
      }

      batfish = BatfishHelper.getBatfishFromTestrigText(containerPath, fullConfigs);
      settings = batfish.getSettings();
      scenarioPath = Paths.get(path).getParent();

      configurations = batfish.loadConfigurations();

      numDataplanes = 0;

      graph = new org.batfish.symbolic.Graph(batfish);

      answer = "Success";
    } else {
      answer = "Failure";
    }

    return answer;
  }


  private String processQuery(String req) {

    long startTime = System.currentTimeMillis();

    HeaderLocationQuestion question = mapRequestToQuestion(req);

    PropertyChecker p = new PropertyChecker(new BDDPacket(), batfish, settings);

    String answer = "";
    if (question instanceof ReachabilityQuestion) {
      answer = (p.checkReachability(question)).prettyPrint();
    } else if (question instanceof NumPathsQuestion) {
      int k = ((NumPathsQuestion) question).getNumPaths();
      answer = (p.checkLoadBalancingSimple(question, k)).prettyPrint();
    } else if (question instanceof WaypointsQuestion) {
      List<String> waypoints = ((WaypointsQuestion) question).getWaypoints();
      answer = (p.checkWaypoints(question, waypoints)).prettyPrint();
    }

    return answer;
  }

  public String processQuery(Request req, Response res) {

    long startTime = System.currentTimeMillis();

    HeaderLocationQuestion question = mapRequestToQuestion(req.body());

    if (!verifyRecursiveHotEdges(question)){
      return "Not held";
    }

    PropertyChecker p = new PropertyChecker(new BDDPacket(), batfish, settings);

    String answer = "";
    if (question instanceof ReachabilityQuestion) {
      answer = (p.checkReachability(question)).prettyPrint();
    } else if (question instanceof NumPathsQuestion) {
      int k = ((NumPathsQuestion) question).getNumPaths();
      answer = (p.checkLoadBalancingSimple(question, k)).prettyPrint();
    } else if (question instanceof WaypointsQuestion) {
      List<String> waypoints = ((WaypointsQuestion) question).getWaypoints();
      answer = (p.checkWaypoints(question, waypoints)).prettyPrint();
    }
  
    return answer;
  }

  public String getTopology(Request req, Response res) {

    Topology topology = batfish.getEnvironmentTopology();

    String fileNameTopology = "topology.txt";
    Path fileTopology = scenarioPath.resolve(fileNameTopology);
    File f1 = new File(fileTopology.toString());

    SortedSet<Edge> tmpEdges = topology.getEdges();

    ArrayList<String> edges = new ArrayList<>();
    for (Edge edge: tmpEdges) {
      edges.add(edge.toString());
    }

    try {
      f1.createNewFile();
      Files.write(fileTopology, edges, Charset.forName("UTF-8"));
    } catch (IOException e) {
      System.out.println("IOException when trying to create and write to the file: " + fileNameTopology + "; " + e);
    }

    String fileNameInterfaces = "interfaces.txt";
    Path fileInterfaces = scenarioPath.resolve(fileNameInterfaces);
    File f2 = new File(fileInterfaces.toString());

    ArrayList<String> interfaceEntries = new ArrayList<String>();
    Map<String, SortedSet<Edge>> nodeEdges = topology.getNodeEdges();
    for (String node : nodeEdges.keySet()) {
      interfaceEntries.add("# Router:" + node);
      Configuration routerConfig = configurations.get(node);

      Map<String, Interface> interfaces = routerConfig.getAllInterfaces();
      for (Map.Entry<String, Interface> entry : interfaces.entrySet()) {
        String intfName = entry.getKey();

        StringBuilder intfSb = new StringBuilder();

        Interface intf = entry.getValue();

        intfSb.append("## Interface:" + intfName);

        IpAccessList inFilter = intf.getIncomingFilter();
        if (inFilter != null) {
          intfSb.append(";IN:" + inFilter.getName());
        }

        IpAccessList outFilter = intf.getOutgoingFilter();
        if (outFilter != null) {
          intfSb.append(";OUT:" + outFilter.getName());
        }

        interfaceEntries.add(intfSb.toString());

        Set<InterfaceAddress> intfAdresses= intf.getAllAddresses();
        for (InterfaceAddress intfAddress : intfAdresses) {
          interfaceEntries.add(intfAddress.toString());
        }
      }
    }

    try {
      f2.createNewFile();
      Files.write(fileInterfaces, interfaceEntries, Charset.forName("UTF-8"));
    } catch (IOException e) {
      System.out.println(
          "IOException when trying to create and write to the file: " + fileNameInterfaces + "; " + e);
    }

    String fileNameAcl = "acls.txt";
    Path fileAcl = scenarioPath.resolve(fileNameAcl);
    File f3 = new File(fileAcl.toString());

    ArrayList<String> aclEntries = new ArrayList<String>();
    for (String node : nodeEdges.keySet()) {
      aclEntries.add("# Router:" + node);
      Configuration routerConfig = configurations.get(node);

      Map<String, IpAccessList> aclDefinitions = routerConfig.getIpAccessLists();
      for (Map.Entry<String, IpAccessList> entry : aclDefinitions.entrySet()) {
        String aclName = entry.getKey();
        IpAccessList acl = entry.getValue();
        List<IpAccessListLine> aclLines = acl.getLines();
        for (IpAccessListLine aclLine : aclLines) {
          aclEntries.add(aclName + ":" + aclLine.getName());
        }
      }
    }

    try {
      f3.createNewFile();
      Files.write(fileAcl, aclEntries, Charset.forName("UTF-8"));
    } catch (IOException e) {
      System.out.println("IOException when trying to create and write to the file: " + fileNameAcl + "; " + e);
    }

    return "TOPO:" + fileNameTopology + ";INTERFACES:" + fileNameInterfaces + ";ACL:" + fileNameAcl;
  }

  public String getDataplane(Request req, Response res) {

    ++numDataplanes;

    Path fibPath = scenarioPath.resolve("fibs");
    new File(fibPath.toString()).mkdirs();

    String fileNameFib = "fib-" + numDataplanes + ".txt";

    Path fileFib = fibPath.resolve(fileNameFib);
    File f = new File(fileFib.toString());
    try {
      f.createNewFile();
      Files.write(fileFib, new ArrayList<String>(), Charset.forName("UTF-8"));
    } catch (IOException e) {
      System.out.println("IOException when trying to create the file: " + fileNameFib + "; " + e);
    }

    Topology topology = batfish.getEnvironmentTopology();

    Map<String, SortedSet<Edge>> nodeEdges = topology.getNodeEdges();

    String request = req.body();

    TreeSet<Edge> blackListEdges = new TreeSet<Edge>();
    if (request.contains("EdgeBlacklist:")) {
      String[] envParameters = request.split(":");

      if (envParameters.length > 1) {
        String[] edges = envParameters[1].split(",");

        for (String edge : edges) {

          String[] nodes = edge.split("=");
          Set<Edge> tmpEdges1 = nodeEdges.get(nodes[0]);
          Set<Edge> tmpEdges2 = nodeEdges.get(nodes[1]);

          TreeSet<Edge> intersection = new TreeSet<Edge>(tmpEdges1);
          intersection.retainAll(tmpEdges2);

          blackListEdges.addAll(intersection);
        }
      }

      topology.prune(blackListEdges, null, null);
    }

    DataPlane dp = null;
    try {
      ComputeDataPlaneResult dpResult = batfish.getDataPlanePlugin().computeDataPlane(false, configurations, topology);
      dp = dpResult._dataPlane;
    } catch (BdpOscillationException e) {
      System.out.println(e + " for the following blacklisted edges: " + blackListEdges);
    }

    if (dp != null) {
      Map<String, Map<String, Fib>> fibsData = dp.getFibs();

      for (Map.Entry<String, Map<String, Fib>> entry: fibsData.entrySet()) {
        String router = entry.getKey();

        ArrayList<String> fibEntries = new ArrayList<String>();
        fibEntries.add("# Router:" + router);

        for (Map.Entry<String, Fib> entry2: entry.getValue().entrySet()) {
          String vrf = entry2.getKey();
          fibEntries.add("## VRF:" + vrf);

          Fib tmpFib = entry2.getValue();

          for (Map.Entry<String, Set<AbstractRoute>> fibEntry: tmpFib.getRoutesByNextHopInterface().entrySet()) {
            String nextHopInterface = fibEntry.getKey();

            for (AbstractRoute route : fibEntry.getValue()) {
              if (!(route instanceof LocalRoute)) {
                // TODO consider ACLs!!!!!
                fibEntries.add(route.getNetwork() + ";" + nextHopInterface + ";" + route.getClass().getSimpleName());
              }
            }
          }
        }
        try {
          Files.write(fileFib, fibEntries, Charset.forName("UTF-8"), StandardOpenOption.APPEND);
        } catch (IOException e) {
          System.out.println("IOException when trying to write to file: " + fileNameFib + "; " + e);
        }
      }
    } else {
      return "ERROR";
    }

    return "FIB:" + fileNameFib;
  }

  private HeaderLocationQuestion mapRequestToQuestion(String request) {
    List<String> queryParameters = Arrays.asList(request.split(";"));

    HeaderLocationQuestion question = null;

    if (request.contains("Type:reachability")) {
      question = new ReachabilityQuestion();
    } else if (request.contains("Type:loadbalancing")) {
      question = new NumPathsQuestion();
    } else if (request.contains("Type:waypoint")) {
      question = new WaypointsQuestion();
    } else {
      System.out.println("Unknown Question Type: " + request);
      return null;
    }

    String source = null, destination = null, prefix = null, fullLinkPaths = "";
    int depth = 1, maxFailures = 0;


    for (String parameter : queryParameters) {
      String[] item = parameter.split(":");
      if (item.length > 1) {

        switch (item[0]) {
        case "Negate":
          question.setNegate(Boolean.parseBoolean(item[1]));
          break;
        case "IngressNodeRegex":
          question.setIngressNodeRegex(item[1]);
          source = item[1].replace("^", "").replace("$","");
          break;
        case "FinalNodeRegex":
          question.setFinalNodeRegex(item[1]);
          destination = item[1];
          break;
        case "FinalIfaceRegex":
          question.setFinalIfaceRegex(item[1]);
          break;
        case "MaxFailures":
          question.setFailures(Integer.parseInt(item[1]));
          maxFailures = Integer.parseInt(item[1]);
          break;
        case "Environment":
          question.setFailureEnvironment(item[1]);
          break;
        case "FullLinkPaths":
          question.setFullLinkPath(item[1]);
          fullLinkPaths = item[1];
          break;
        case "Depth":
           depth = Integer.parseInt(item[1]);
           question.setDepth(depth);
          break;
        case "Prefix":
          prefix = item[1];
          break;
        case "Waypoints":
          if (question instanceof WaypointsQuestion) {
            ArrayList<String> waypointList = new ArrayList<>(Arrays.asList(item[1].split(",")));

            Collections.reverse(waypointList);
            ((WaypointsQuestion) question).setWaypoints(waypointList);
          }
          break;
        case "NumPaths":
          if (question instanceof NumPathsQuestion) {
            ((NumPathsQuestion) question).setNumPaths(Integer.parseInt(item[1]));
          }
          break;
        case "Type":
          break;
        default:
          System.out.println("Unknown Parameter: " + item[0] + " - " + item[1]);
        }
      }
    }

    if (depth > 0)
      question.setBestPathWithBrokenEdges(getRecursiveHotEdges(depth, maxFailures, prefix, source, destination, fullLinkPaths));
    return question;
  }


  private String getRecursiveHotEdges(int recursiveDepth, int maxFailures, String prefixStr,
      String source, String destination, String fullLinkPath){

    if (fullLinkPath.equals(""))
      return "";

    Topology topology = batfish.getEnvironmentTopology();

    int depth = Math.min(recursiveDepth, maxFailures);
    Prefix prefix = Prefix.parse(prefixStr);

    StringBuilder bestLinks = new StringBuilder();

    String [] hotNodes = fullLinkPath.split("-");
    for(int i=0; i<hotNodes.length-1; i++){
      bestLinks.append(hotNodes[i]).append("=").append(hotNodes[i+1]).append(",");
    }
    bestLinks.deleteCharAt(bestLinks.length()-1);


    long startTime = System.currentTimeMillis();

    Test(bestLinks.toString(), topology, depth, new ArrayList<>(), prefix, source, destination);

    long duration = System.currentTimeMillis() - startTime;

    StringBuilder sb = new StringBuilder();

    for(Tuple<String, String> tup: brokenHotEdgeList){
      if (!tup.getFirst().equals("")){
        sb.append(tup.getFirst());
        sb.append(":");
      }
      sb.append(tup.getSecond());
      sb.append(";");
    }

    brokenHotEdgeList.clear();

    sb.deleteCharAt(sb.length()-1);
    return sb.toString();

  }

  private boolean verifyRecursiveHotEdges(@Nullable HeaderLocationQuestion question){

    PathRegexes p = new PathRegexes(question);
    Set<GraphEdge> destPorts = findFinalInterfaces(graph, p);
    List<String> sourceRouters = PatternUtils.findMatchingSourceNodes(graph, p);
    inferDestinationHeaderSpace(graph, destPorts, question);

    Set<Prefix> prefixes = new HashSet<>();
    question.getDstIps().forEach(ipWildcard -> {
      if (ipWildcard.isPrefix()){
        prefixes.add(ipWildcard.toPrefix());
      }else{
        prefixes.add(Prefix.parse(ipWildcard.getIp().toString()));
      }
    });

    if (destPorts.isEmpty()) {
      throw new BatfishException("Set of valid destination interfaces is empty");
    }
    if (sourceRouters.isEmpty()) {
      throw new BatfishException("Set of valid ingress nodes is empty");
    }

    assert prefixes.size() == 1;
    assert sourceRouters.size() == 1;

    String srcRouter = sourceRouters.get(0);

    String bestPathWithBrokenEdges = question.getBestPathWithBrokenEdges();
    String [] derivedBestPathPairs =  bestPathWithBrokenEdges.split(";");

    if (question instanceof ReachabilityQuestion){
      for (String derivedBestPathPair : derivedBestPathPairs) {
        String[] pathPair = derivedBestPathPair.split(":");

        if (pathPair.length != 2)
          return true;
        assert pathPair.length == 2;

        String[] brokenEdges = pathPair[0].split(",");
        String derivedBestPath = pathPair[1];

        if (!derivedBestPath.contains(srcRouter)){
          return false;
        }
      }
      return true;
    } else if (question instanceof WaypointsQuestion){
      List<String> waypoints = ((WaypointsQuestion) question).getWaypoints();

      for (String derivedBestPathPair : derivedBestPathPairs) {
        String[] pathPair = derivedBestPathPair.split(":");

        if (pathPair.length != 2)
          return true;

        assert pathPair.length == 2;
        String[] brokenEdges = pathPair[0].split(",");
        String derivedBestPath = pathPair[1];

        for (String waypoint : waypoints) {
          if (!derivedBestPath.contains(waypoint)){
            return false;
          }
        }
      }
      return true;
    } else if (question instanceof NumPathsQuestion){
      int k = ((NumPathsQuestion) question).getNumPaths();

      for (String derivedBestPathPair : derivedBestPathPairs) {
        String[] pathPair = derivedBestPathPair.split(":");
        if (pathPair.length != 2)
          return true;

        assert pathPair.length == 2;
        String[] brokenEdges = pathPair[0].split(",");
        String derivedBestPath = pathPair[1];

        Graph<String, DefaultEdge> tmpGraph = new DefaultDirectedGraph<>(DefaultEdge.class);

        String [] newHotEdges = derivedBestPath.split(",");
        for (String newHotEdge: newHotEdges){
          String router = newHotEdge.split("=")[0];
          String peer = newHotEdge.split("=")[1];
          tmpGraph.addVertex(router);
          tmpGraph.addVertex(peer);
          tmpGraph.addEdge(router, peer);
        }

        String dst = null;
        for (String s : tmpGraph.vertexSet()) {
          if (tmpGraph.outDegreeOf(s)==0)
            dst = s;
        }
        List<GraphPath<String, DefaultEdge>> paths = new AllDirectedPaths<String, DefaultEdge>(tmpGraph).getAllPaths(srcRouter, dst, true, 300);
        if (paths.size() < k)
          return false;
      }

      return true;
    } else{
      throw new BatfishException("This kind of policy is not implemented yet");
    }
  }

  private String getSingleHotEdges(Topology topology, Prefix prefix, String source,
      String destination){

    ComputeDataPlaneResult dpResult = batfish.getDataPlanePlugin().computeDataPlane(false, configurations, topology);
    DataPlane dp = dpResult._dataPlane;

    Map<String, Map<String, Map<String, Fib>>> auxFibsData = new HashMap<>();
    auxFibsData.put("all", dp.getFibs());

    Graph<String, DefaultEdge> fwdGraph = buildForwardingGraph(auxFibsData.get("all"), prefix, topology);

    Set<DefaultEdge> hotEdges = new HashSet<>();

    Queue<String> queue = new ArrayDeque<>();
    queue.add(source);
    while(!queue.isEmpty()){
      String top = queue.poll();
      for(DefaultEdge edge: fwdGraph.outgoingEdgesOf(top)){
        hotEdges.add(edge);
        if(queue.contains(fwdGraph.getEdgeTarget(edge)))
          continue;
        queue.add(fwdGraph.getEdgeTarget(edge));
      }
    }

    StringBuilder sb = new StringBuilder();

    hotEdges.stream().forEach(edge -> sb.append(fwdGraph.getEdgeSource(edge))
        .append("=")
        .append(fwdGraph.getEdgeTarget(edge))
        .append(","));

    sb.deleteCharAt(sb.length()-1);

    return sb.toString();
  }

  private Graph<String, DefaultEdge> buildForwardingGraph(Map<String, Map<String, Fib>> fibs,
      Prefix prefix, Topology topology){

    Graph<String, DefaultEdge> fwdGraph = new DefaultDirectedGraph<>(DefaultEdge.class);
    Map<String, SortedSet<Edge>> nodeEdges = topology.getNodeEdges();
    nodeEdges.keySet().forEach(fwdGraph::addVertex);

    for (String router : nodeEdges.keySet()) {

      Fib routerFib = fibs.get(router).get("default");

      for(Map.Entry<String, Set<AbstractRoute>> fibEntry:
          routerFib.getRoutesByNextHopInterface().entrySet()){
        for (AbstractRoute route: fibEntry.getValue()){
          if (!prefix.containsPrefix(route.getNetwork()))
            continue;
          NodeInterfacePair nipair = new NodeInterfacePair(router, fibEntry.getKey());

          for (NodeInterfacePair pair : topology.getNeighbors(nipair)) {
            fwdGraph.addEdge(router, pair.getHostname());
          }
        }
      }
    }
    return fwdGraph;
  }


  private Set<GraphEdge> findFinalInterfaces(org.batfish.symbolic.Graph g, PathRegexes p) {
    Set<GraphEdge> edges = new HashSet<>(PatternUtils.findMatchingEdges(g, p));
    return edges;
  }

  private void inferDestinationHeaderSpace(org.batfish.symbolic.Graph g,
      Collection<GraphEdge> destPorts, HeaderLocationQuestion q) {
    if (q.getHeaderSpace().getDstIps() != null
        && q.getHeaderSpace().getDstIps() != EmptyIpSpace.INSTANCE) {
      return;
    }

    HeaderSpace headerSpace = q.getHeaderSpace();
    for (GraphEdge ge : destPorts) {

      if (g.isExternal(ge)) {
        headerSpace.setDstIps(Collections.emptySet());
        headerSpace.setNotDstIps(Collections.emptySet());
        break;
      }
      if (ge.getPeer() == null) {
        Prefix pfx = ge.getStart().getAddress().getPrefix();
        IpWildcard dst = new IpWildcard(pfx);
        headerSpace.setDstIps(AclIpSpace.union(headerSpace.getDstIps(), dst.toIpSpace()));
      } else {
        if (g.isHost(ge.getRouter())) {
          Prefix pfx = ge.getStart().getAddress().getPrefix();
          IpWildcard dst = new IpWildcard(pfx);
          headerSpace.setDstIps(AclIpSpace.union(headerSpace.getDstIps(), dst.toIpSpace()));
          Ip ip = ge.getEnd().getAddress().getIp();
          IpWildcard dst2 = new IpWildcard(ip);
          headerSpace.setNotDstIps(AclIpSpace.union(headerSpace.getNotDstIps(), dst2.toIpSpace()));
        } else {
          Ip ip = ge.getStart().getAddress().getIp();
          IpWildcard dst = new IpWildcard(ip);
          headerSpace.setDstIps(AclIpSpace.union(headerSpace.getDstIps(), dst.toIpSpace()));
        }
      }
    }
  }

}
