package org.batfish.backend;

import static spark.Spark.port;
import static spark.Spark.post;

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
import org.batfish.symbolic.utils.Tuple;
import org.batfish.symbolic.utils.PathRegexes;
import org.batfish.symbolic.utils.PatternUtils;
import org.jgrapht.Graph;
import org.jgrapht.GraphPath;
import org.jgrapht.alg.shortestpath.AllDirectedPaths;
import org.jgrapht.graph.DefaultDirectedGraph;
import org.jgrapht.graph.DefaultEdge;
import spark.Request;
import spark.Response;



public class Backend {

  public static void main(String[] args) {
   if (args.length > 0) {
     port(Integer.parseInt(args[0]));
   } else {
     port(8192);
   }

   Server s = new Server();


   // Configure Spark
   post("/run_query", (req, res) -> s.processQuery(req, res));

   post("/init_minesweeper", (req, res) -> s.initMinesweeper(req, res));

   post("/get_dataplane", (req, res) -> s.getDataplane(req, res));

   post("/get_topology", (req, res) -> s.getTopology(req, res));

  }
}




