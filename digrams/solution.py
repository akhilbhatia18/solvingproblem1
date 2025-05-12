from diagrams import Diagram, Cluster,Edge
from diagrams.aws.network import CloudFrontEdgeLocation
from diagrams.aws.ml import Sagemaker
from diagrams.onprem.compute import Server
from diagrams.programming.language import Python
from diagrams.onprem.client import Users
from diagrams.generic.storage import Storage
from diagrams.generic.network import Router

with Diagram("End to End Framework", show=False):
    # User Devices
    users = Users("User Devices")

    # Edge Node Deployment
    with Cluster("Edge Node Deployment"):
        kmeans = Python("KMeans Clustering")
        edge_nodes = Router("Edge Nodes")
    with Cluster("GASO Framework"):
    # Load Forecasting
        with Cluster("Load Forecasting"):
            vmd = Sagemaker("VMD Decomposition")
            gru = Sagemaker("GRU Model")
            forecast = Server("Forecasted Load")
        # Task Offloading
        with Cluster("Task Offloading"):
            stackelberg = CloudFrontEdgeLocation("Stackelberg Game")
            task_assignment = Server("Task Assignment")

        # Service Migration
        with Cluster("Service Migration"):
            tigo_stages = CloudFrontEdgeLocation("TIGO Stage 2")
            migration_decision = Server("Migration Decision")

        with Cluster("Resource Orchestration"):
            migration_decision = Server("Energy Metrics")

    # Data Storage
    storage = Storage("Results")

    # Workflow Connections
    users >> kmeans >> Edge(color="black", style="dashed") >> edge_nodes
    edge_nodes >> vmd >> Edge(color="blue", style="dashed") >> gru >> Edge(color="blue", style="dashed") >> forecast
    forecast >> stackelberg >> Edge(color="red", style="dashed") >> task_assignment
    task_assignment >> tigo_stages >> Edge(color="orange", style="dashed") >> migration_decision >> Edge(color="black", style="bold") >> vmd
    migration_decision >> storage