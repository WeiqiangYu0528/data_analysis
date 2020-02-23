     function escape2Html(str) {
     let arrEntities={'lt':'<','gt':'>','nbsp':' ','amp':'&','quot':'"'};
     return str.replace(/&(lt|gt|nbsp|amp|quot);/ig,function(all,t){return arrEntities[t];});
    }

     function drawd3(p1,p2) {
         d3.select("#force").selectAll("*").remove();
         d3.select("#force")
             .append("svg")
             .attr("id","svg")
             .attr("width",800)
             .attr("height",600)

         let svg = d3.select("#svg"),
         width = +svg.attr("width"),
         height = +svg.attr("height");

         let color = d3.scaleOrdinal(d3.schemeCategory20);

         let g = svg.append("g")
        .attr("class", "everything");

         let simulation = d3.forceSimulation()
             .force("link", d3.forceLink().id(function (d) {
                 return d.id;
             }))
             .force("charge", d3.forceManyBody().strength(-50))
             .force("collide", d3.forceCollide(10).strength(0.9))
             .force("center", d3.forceCenter(width / 2, height / 2));
         
            let part1 = escape2Html(p1)
            let part2 = escape2Html(p2)

         nodes = $.parseJSON(part1)
         links = $.parseJSON(part2)


         let link = g.append("g")
             .attr("class", "links")
             .selectAll("line")
             .data(links)
             .enter().append("line")
             .attr("stroke-width", '1.5')
             .attr("stroke","lightsteelblue");
// function(d) { return Math.sqrt(d.value);


         let node = g.append("g")
             .attr("class", "nodes")
             .selectAll("g")
             .data(nodes)
             .enter().append("g")

         let circles = node.append("circle")
             .attr("r", 5)
             .attr("fill", function (d) {
                 return color(d.group);
             })
             .call(d3.drag()
                 .on("start", dragstarted)
                 .on("drag", dragged)
                 .on("end", dragended));

         let lables = node.append("text")
             .text(function (d) {
                 return d.id;
             })
             .attr('x', 6)
             .attr('y', 3);

         node.append("title")
             .text(function (d) {
                 return d.id;
             });

         simulation
             .nodes(nodes)
             .on("tick", ticked);

         simulation.force("link")
             .links(links);

         let zoom_handler = d3.zoom()
            .on("zoom", zoom_actions);
            zoom_handler(svg);

         let legend = svg.selectAll(".legend")
            .data(color.domain())
            .enter().append("g")
            .attr("class", "legend")
            .attr("transform", function(d, i) { return "translate(0," + i * 20 + ")"; });

        legend.append("rect")
            .attr("x", width - 18)
            .attr("width", 18)
            .attr("height", 18)
            .style("fill", color);

        legend.append("text")
            .attr("x", width - 24)
            .attr("y", 9)
            .attr("dy", ".35em")
            .style("text-anchor", "end")
            .text(function(d) { return d; });

         function ticked() {
             link
                 .attr("x1", function (d) {
                     return d.source.x;
                 })
                 .attr("y1", function (d) {
                     return d.source.y;
                 })
                 .attr("x2", function (d) {
                     return d.target.x;
                 })
                 .attr("y2", function (d) {
                     return d.target.y;
                 });

             node
                 .attr("transform", function (d) {
                     return "translate(" + d.x + "," + d.y + ")";
                 })
         }


         function dragstarted(d) {
             if (!d3.event.active) simulation.alphaTarget(0.3).restart();
             d.fx = d.x;
             d.fy = d.y;
         }

         function dragged(d) {
             d.fx = d3.event.x;
             d.fy = d3.event.y;
         }

         function dragended(d) {
             if (!d3.event.active) simulation.alphaTarget(0);
             d.fx = null;
             d.fy = null;
         }

         function zoom_actions(){
             g.attr("transform", d3.event.transform)
            }
     }



