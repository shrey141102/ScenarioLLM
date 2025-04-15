"""
Converter to transform LLM-generated scenarios into SUMO format.
"""

import os
import re
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom


class SUMOConverter:
    def __init__(self, output_dir='output/sumo_files'):
        """
        Initialize the SUMO converter.

        Args:
            output_dir (str): Directory to save SUMO files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def parse_trajectory_data(self, scenario_text):
        """
        Parse vehicle trajectory data from LLM-generated scenario.

        Args:
            scenario_text (str): LLM-generated scenario text

        Returns:
            dict: Dictionary of vehicle trajectories
        """
        # Extract trajectory data using regex
        vehicle_pattern = r"Vehicle ID: (\d+).*?\n(Time \(s\), X \(m\), Y \(m\), Velocity \(m/s\), Heading \(rad\)\n(?:\d+\.\d+, -?\d+\.\d+, -?\d+\.\d+, -?\d+\.\d+, -?\d+\.\d+\n)+)"
        vehicle_matches = re.findall(vehicle_pattern, scenario_text, re.DOTALL)

        trajectories = {}

        for vehicle_id, trajectory_text in vehicle_matches:
            # Extract each line of trajectory data
            trajectory_lines = trajectory_text.strip().split('\n')[1:]  # Skip header

            vehicle_trajectory = []
            for line in trajectory_lines:
                try:
                    time, x, y, velocity, heading = map(float, line.split(', '))
                    vehicle_trajectory.append({
                        'time': time,
                        'x': x,
                        'y': y,
                        'velocity': velocity,
                        'heading': heading
                    })
                except ValueError:
                    continue  # Skip invalid lines

            if vehicle_trajectory:  # Only add if we have valid trajectory points
                trajectories[vehicle_id] = vehicle_trajectory

        return trajectories

    def create_route_file(self, trajectories, scenario_name):
        """
        Create SUMO route file from trajectories.

        Args:
            trajectories (dict): Dictionary of vehicle trajectories
            scenario_name (str): Name of the scenario

        Returns:
            str: Path to the created route file
        """
        route_file = os.path.join(self.output_dir, f"{scenario_name}.rou.xml")

        # Create XML structure
        routes = ET.Element("routes")

        # Define vehicle types
        vtype = ET.SubElement(routes, "vtype", id="car", accel="2.6", decel="4.5",
                              sigma="0.5", length="5.0", minGap="2.5", maxSpeed="50.0",
                              color="1,1,0")

        # Add vehicles and their routes
        for vehicle_id, trajectory in trajectories.items():
            # Create a route for the vehicle
            start_point = trajectory[0]
            end_point = trajectory[-1]

            # Create a vehicle with the route
            vehicle = ET.SubElement(routes, "vehicle", id=f"vehicle_{vehicle_id}",
                                    type="car", depart=f"{start_point['time']}",
                                    departPos="0", departSpeed=f"{start_point['velocity']}")

            # Add route
            route = ET.SubElement(vehicle, "route", edges="e1")

            # Add trajectory as a series of stops (this is a simplification)
            for i, point in enumerate(trajectory[1:], 1):  # Skip the first point
                time_diff = point['time'] - trajectory[i - 1]['time']
                if time_diff > 0:
                    stop = ET.SubElement(vehicle, "stop", lane="e1_0", endPos=f"{point['x']}",
                                         duration="0", until=f"{point['time']}")

        # Write the XML to a file
        rough_string = ET.tostring(routes, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        with open(route_file, 'w') as f:
            f.write(reparsed.toprettyxml(indent="  "))

        return route_file

    def create_network_file(self, trajectories, scenario_name):
        """
        Create SUMO network file based on trajectories.

        Args:
            trajectories (dict): Dictionary of vehicle trajectories
            scenario_name (str): Name of the scenario

        Returns:
            str: Path to the created network file
        """
        network_file = os.path.join(self.output_dir, f"{scenario_name}.net.xml")

        # Determine network bounds based on trajectories
        x_values = []
        y_values = []

        for vehicle_id, trajectory in trajectories.items():
            for point in trajectory:
                x_values.append(point['x'])
                y_values.append(point['y'])

        min_x = min(x_values) - 100 if x_values else 0
        max_x = max(x_values) + 100 if x_values else 1000
        min_y = min(y_values) - 50 if y_values else 0
        max_y = max(y_values) + 50 if y_values else 100

        # Create XML structure
        net = ET.Element("net", version="1.9", junctionCornerDetail="5",
                         limitTurnSpeed="5.5", xmlns="http://sumo.dlr.de/xsd/net_file.xsd")

        # Add locations
        location = ET.SubElement(net, "location", netOffset="0.00,0.00",
                                 convBoundary=f"{min_x},{min_y},{max_x},{max_y}",
                                 origBoundary=f"{min_x},{min_y},{max_x},{max_y}",
                                 projParameter="!")

        # Add edges (simplified)
        edge = ET.SubElement(net, "edge",
                             attrib={"id": "e1", "from": "j1", "to": "j2", "priority": "1", "numLanes": "4",
                                     "speed": "50.00"})

        # Add lanes (simplified)
        for i in range(4):
            lane = ET.SubElement(edge, "lane", index=str(i), speed="50.00", length=f"{max_x - min_x}",
                                 shape=f"{min_x},{min_y + 5 + i * 5} {max_x},{min_y + 5 + i * 5}")

        # Write the XML to a file
        rough_string = ET.tostring(net, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        with open(network_file, 'w') as f:
            f.write(reparsed.toprettyxml(indent="  "))

        return network_file

    def create_config_file(self, route_file, network_file, scenario_name):
        """
        Create SUMO configuration file.

        Args:
            route_file (str): Path to route file
            network_file (str): Path to network file
            scenario_name (str): Name of the scenario

        Returns:
            str: Path to the created config file
        """
        config_file = os.path.join(self.output_dir, f"{scenario_name}.sumocfg")

        # Create XML structure
        configuration = ET.Element("configuration")

        input_section = ET.SubElement(configuration, "input")
        ET.SubElement(input_section, "net-file", value=os.path.basename(network_file))
        ET.SubElement(input_section, "route-files", value=os.path.basename(route_file))

        time_section = ET.SubElement(configuration, "time")
        ET.SubElement(time_section, "begin", value="0")
        ET.SubElement(time_section, "end", value="100")

        # Remove the gui_section with gui-settings reference

        # Write the XML to a file
        rough_string = ET.tostring(configuration, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        with open(config_file, 'w') as f:
            f.write(reparsed.toprettyxml(indent="  "))

        return config_file

    def convert(self, scenario_text, scenario_name):
        """
        Convert LLM-generated scenario to SUMO files.

        Args:
            scenario_text (str): LLM-generated scenario text
            scenario_name (str): Name of the scenario

        Returns:
            str: Path to the SUMO config file
        """
        # Parse trajectories
        trajectories = self.parse_trajectory_data(scenario_text)

        if not trajectories:
            raise ValueError("No valid trajectory data found in the scenario text")

        # Create network file
        network_file = self.create_network_file(trajectories, scenario_name)

        # Create route file
        route_file = self.create_route_file(trajectories, scenario_name)

        # Create config file
        config_file = self.create_config_file(route_file, network_file, scenario_name)

        return config_file