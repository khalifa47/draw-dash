"use server";

import ApplicationSingleton from "@/app/app";
import {
  type PreTrainedModel,
  type Processor,
  RawImage,
} from "@xenova/transformers";

const cosineSimilarity = (
  query_embeds: Float32Array,
  ans_embeds: Float32Array
) => {
  if (query_embeds.length !== ans_embeds.length) {
    throw new Error("Embeddings must be of the same length");
  }

  let dotProduct = 0;
  let normQuery = 0;
  let normAns = 0;

  for (let i = 0; i < query_embeds.length; ++i) {
    const queryValue = query_embeds[i];
    const dbValue = ans_embeds[i];

    dotProduct += queryValue * dbValue;
    normQuery += queryValue * queryValue;
    normAns += dbValue * dbValue;
  }

  const similarity = dotProduct / (Math.sqrt(normQuery) * Math.sqrt(normAns));
  return similarity;
};

export async function compare(query_image: string, ans_image: string) {
  const [processor, vision_model]: [Processor, PreTrainedModel] =
    await ApplicationSingleton.getInstance();

  // Read the two images as raw images
  const rawImageQuery = await RawImage.fromURL(query_image);
  const rawImageAns = await RawImage.fromURL(ans_image);

  // Tokenize the two images
  const tokenizedImageQuery = await processor(rawImageQuery);
  const tokenizedImageAns = await processor(rawImageAns);

  // Encode the two images
  const { image_embeds: embedsQuery } = await vision_model(tokenizedImageQuery);
  const { image_embeds: embedsAns } = await vision_model(tokenizedImageAns);

  // Directly convert the embeddings to Float32Array for memory efficiency
  const queryEmbedsArray = new Float32Array(
    Object.values(embedsQuery.data) as number[]
  );
  const ansEmbedsArray = new Float32Array(
    Object.values(embedsAns.data) as number[]
  );

  const similarity = cosineSimilarity(queryEmbedsArray, ansEmbedsArray);

  return similarity;
}
