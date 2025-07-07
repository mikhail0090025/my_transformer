from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
import my_transformer as MT
import matplotlib.pyplot as plt

app = FastAPI()

@app.get("/")
async def go_epochs(epochs: int = 1):
    return Response("This is a root")

@app.get("/train/{epochs}")
async def go_epochs(epochs: int = 1):
    print("Go ", epochs, " epochs")
    train_loss, val_loss = MT.go_epochs(epochs, MT.my_trans, MT.train_dataset_loader, MT.val_dataset_loader, MT.optimizer)
    return JSONResponse({
        'train_loss': train_loss,
        'val_loss': val_loss,
    })

@app.get("/graph")
async def graph_endpoint():
    plt.plot(range(len(MT.all_val_losses)), MT.all_train_losses, marker='o', linestyle='-', color='b', label='Train Loss')
    plt.plot(range(len(MT.all_val_losses)), MT.all_val_losses, marker='o', linestyle='-', color='orange', label='Validation Loss')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Simple Line Plot')
    plt.legend()
    plt.grid(True)
    plt.show()
    return Response("Graph")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)