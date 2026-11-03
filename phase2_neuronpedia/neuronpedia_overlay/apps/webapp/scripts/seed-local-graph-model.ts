// scripts/seed-local-graph-model.ts
import { PrismaClient, Visibility } from '@prisma/client';

const prisma = new PrismaClient();

async function main() {
  const modelId = 'local-gemma-2b-toolformer';
  const sourceSetName = 'toolformer-fused';
  const layers = 26; // set to the model's number of layers

  console.log(`Seeding model ${modelId} with ${layers} layers...`);

  // 0. Create or fetch a local user to use as creatorId
  const localUser = await prisma.user.upsert({
    where: { id: 'local-user-id' },
    update: {},
    create: {
      id: 'local-user-id',
      name: 'Local User',
      email: 'local@example.com',
      emailVerified: null,
      image: null,
    },
  });

  const LOCAL_CREATOR_ID = localUser.id;

  // 1. Create or fetch the Model
  const model = await prisma.model.upsert({
    where: { id: modelId },
    update: {},
    create: {
      id: modelId,
      displayName: 'Local Gemma 2B Toolformer',
      displayNameShort: 'local-gemma2b',
      owner: 'local-user',
      layers,
      neuronsPerLayer: 0,
      visibility: Visibility.PRIVATE,
      creatorId: LOCAL_CREATOR_ID,
    },
  });

  console.log('Model upserted:', model.id);

  // 2. Create or fetch the SourceSet
  const sourceSet = await prisma.sourceSet.upsert({
    where: {
      modelId_name: {
        modelId,
        name: sourceSetName,
      },
    },
    update: {},
    create: {
      modelId,
      name: sourceSetName,
      description: 'Local toolformer fused sources',
      visibility: Visibility.UNLISTED,
      hasDashboards: true,
      creatorId: LOCAL_CREATOR_ID,
      creatorName: 'Local',
    },
  });

  console.log('SourceSet upserted:', sourceSet.name);

  // 3. Create one Source per layer (0..layers-1)
  const sourcesPromises: Promise<any>[] = [];
  for (let layer = 0; layer < layers; layer += 1) {
    const sourceId = `${layer}-${sourceSetName}`;
    sourcesPromises.push(
      prisma.source.upsert({
        where: {
          modelId_id: {
            modelId,
            id: sourceId,
          },
        },
        update: {},
        create: {
          modelId,
          id: sourceId,
          setName: sourceSetName,
          creatorId: LOCAL_CREATOR_ID,
          visibility: Visibility.UNLISTED,
          inferenceEnabled: false,
          hasDashboards: true,
        },
      }),
    );
  }

  await Promise.all(sourcesPromises);
  console.log(`Sources upserted for ${layers} layers.`);

  console.log('Done seeding local graph model + source set + sources.');
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });