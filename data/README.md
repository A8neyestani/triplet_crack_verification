# Local triplet data

`triplet_crack_dataset/` contains the local working copy used by the research
notebooks: 20,000 anchor images and 20,000 transformed positive counterparts.
The paper describes the underlying Crack Dataset as a 20,000-image dataset and
reports a 12,000 / 3,000 / 5,000 train-validation-test split after triplet
preparation.

The positive images are generated transformations of anchor images, including
rotation and shearing. They are not longitudinal observations of the same
physical crack. Check the source dataset's terms before redistributing this
directory.
