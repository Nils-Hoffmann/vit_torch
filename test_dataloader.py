import dataloaderv2

# Create a simple pipeline with 1 worker and torch tensor output
pipeline = dataloaderv2.build_cifar10_pipeline(
    batch_size=4,
    augment=True,
    num_workers=1,
    infinite=False,
    to_torch=True
)

print("Getting first batch...")
data_iter = iter(pipeline)
x, y = next(data_iter)

print(f"x shape: {x.shape}, dtype: {x.dtype}, type: {type(x)}")
print(f"y shape: {y.shape}, dtype: {y.dtype}, type: {type(y)}")
print("Success!")
