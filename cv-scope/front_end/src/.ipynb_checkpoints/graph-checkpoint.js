import {useState, useRef, useEffect} from 'react'
//import * as d3 from "d3";
const d3 = require("d3")
function BuildGraph(props) {
  
  console.log('222 ............')
  
  // Build graph data
  //const ref = useRef(null)
  //console.log(props)
  var graph = props.gph;
  var svg   = d3.select("#model_graph")
  
  var width  = 600//svg.attr("width");
  var height = 10000//svg.attr("height");
  // Make the graph scrollable.
  svg = svg.call(d3.zoom().on("zoom", function() {
    svg.attr("transform", d3.event.transform);
  })).append("g");


  var color = d3.scaleOrdinal(d3.schemeDark2);

  var simulation = d3.forceSimulation()
      .force("link", d3.forceLink().id(function(d) {return d.id;}))
      .force("charge", d3.forceManyBody())
      .force("center", d3.forceCenter(0.5 * width, 0.5 * height));

  var edge = svg.append("g").attr("class", "edges").selectAll("line")
    .data(graph.edges).enter().append("path").attr("stroke","black").attr("fill","none")

  // Make the node group
  var node = svg.selectAll(".nodes")
    .data(graph.nodes)
    .enter().append("g")
    .attr("x", function(d){return d.x})
    .attr("y", function(d){return d.y})
    .attr("transform", function(d) {
      return "translate( " + d.x + ", " + d.y + ")"
    })
    .attr("class", "nodes")
      .call(d3.drag()
          .on("start", function(d) {
            if(!d3.event.active) simulation.alphaTarget(1.0).restart();
            d.fx = d.x;d.fy = d.y;
          })
          .on("drag", function(d) {
            d.fx = d3.event.x; d.fy = d3.event.y;
          })
          .on("end", function(d) {
            if (!d3.event.active) simulation.alphaTarget(0);
            d.fx = d.fy = null;
          }));
  // Within the group, draw a box for the node position and text
  // on the side.

  var node_width = 150;
  var node_height = 30;

  node.append("rect")
      .attr("r", "5px")
      .attr("width", node_width)
      .attr("height", node_height)
      .attr("rx", function(d) { return d.group == 1 ? 1 : 10; })
      .attr("stroke", "#000000")
      .attr("fill", function(d) { return d.group == 1 ? "#dddddd" : "#000000"; })
  node.append("text")
      .text(function(d) { return d.name; })
      .attr("x", 5)
      .attr("y", 20)
      .attr("fill", function(d) { return d.group == 1 ? "#000000" : "#eeeeee"; })
  // Setup force parameters and update position callback


  var node = svg.selectAll(".nodes")
    .data(graph.nodes);

  // Bind the links
  var name_to_g = {}
  node.each(function(data, index, nodes) {
    console.log(data.id)
    name_to_g[data.id] = this;
  });

  function proc(w, t) {
    return parseInt(w.getAttribute(t));
  }
  edge.attr("d", function(d) {
    function lerp(t, a, b) {
      return (1.0-t) * a + t * b;
    }
    var x1 = proc(name_to_g[d.source],"x") + node_width /2;
    var y1 = proc(name_to_g[d.source],"y") + node_height;
    var x2 = proc(name_to_g[d.target],"x") + node_width /2;
    var y2 = proc(name_to_g[d.target],"y");
    var s = "M " + x1 + " " + y1
        + " C " + x1 + " " + lerp(.5, y1, y2)
        + " " + x2 + " " + lerp(.5, y1, y2)
        + " " + x2  + " " + y2
  return s;
});
//
return (<>
        {'ppp'}
        <svg id="model_graph" scroll={'yes'} style={{width:"600", height:"10000"}} ></svg>
        </>)
}

export default BuildGraph